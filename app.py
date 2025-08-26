import os
from openai import OpenAI
from flask import Flask, request, jsonify
from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext
import traceback
import re
import unicodedata

import requests
from bs4 import BeautifulSoup
from llama_index import Document  # Ya estás importando VectorStoreIndex, te falta Document

# ============= CONFIGURACIÓN =============
CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "INDEX_NAME": os.getenv("INDEX_NAME"),
    "PINECONE_ENV": os.getenv("PINECONE_ENV"),
    "OPENAI_MODEL": "gpt-3.5-turbo",
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 2000,
    "GOOGLE_SEARCH_API_KEY": os.getenv("GOOGLE_SEARCH_API_KEY"),
    "GOOGLE_CX": os.getenv("GOOGLE_CX")
}


MATH_KEYWORDS = ['calcular', 'cálculo', 'fórmula', 'sumar', 'restar', 'multiplicar', 'dividir']
TOP_K_RESULTS = 25
SIMILARITY_THRESHOLD = 0.5
MAX_ARTICULOS_CON_TEXTO = 5

app = Flask(__name__)

# ============= UTILIDADES =============

def normalizar(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    ).lower()

# ============= LLAMA INDEX =============

pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"], environment=CONFIG["PINECONE_ENV"])
pinecone_index = pc.Index(CONFIG["INDEX_NAME"])

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
)

embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    api_key=CONFIG["OPENAI_API_KEY"]
)

# Cargar el índice de Pinecone donde están los contratos
contratos_index = VectorStoreIndex.from_vector_store(
    PineconeVectorStore(
        pinecone_index=pc.Index("indice-contratos-legales"),
        text_key="text"
    ),
    service_context=ServiceContext.from_defaults(embed_model=embed_model)
)

service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=service_context
)

openai_client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

# ============= RESPUESTA LEGAL =============

def generate_legal_response(question, context_docs, contexto_practico=None):
    system_prompt = """
Eres un abogado especialista en derecho ecuatoriano. Tu tarea es responder EXCLUSIVAMENTE con base en los textos legales entregados a continuación. Está TERMINANTEMENTE PROHIBIDO utilizar conocimiento externo, suposiciones, interpretaciones o completar información más allá de lo provisto.

🧠 Objetivo general:
Redacta una respuesta útil, clara y jurídica que pueda ser comprendida tanto por ciudadanos sin formación legal como por abogados.

🫱 Empatía inicial:
Si la pregunta revela angustia, preocupación o un problema delicado (como cárcel, salud, familia, etc.), comienza con una frase empática y humana, como: “Entendemos lo difícil que puede ser esta situación…” o “Lamentamos lo ocurrido y con gusto le orientamos…”.

📘 Estructura obligatoria:
1. Da una respuesta clara y directa a la pregunta, explicando el contenido legal con palabras sencillas.
2. Cada afirmación debe mencionar de qué artículo y qué código o ley proviene, si aplica.
3. Incluye citas textuales relevantes del texto legal, incluso si están truncadas.
4. Finaliza siempre con la frase: “Me baso en [artículos citados]”.

⚠️ Reglas estrictas:
- NO cites artículos, códigos o leyes que no estén literalmente presentes en el contexto legal proporcionado.
- NO utilices jurisprudencia, doctrina, interpretación propia ni conocimiento externo.
- NO completes ideas que no estén expresamente contenidas en el texto legal.
- Si no hay normativa aplicable, responde exactamente: “No encontré normativa aplicable. No me baso en ningún artículo.”
"""

    context_text = "\nDOCUMENTOS LEGALES:\n" + "\n".join(
        f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto'][:600]}"
        for doc in context_docs
    )

    if contexto_practico:
        context_text += f"\n\n🧾 Contexto práctico adicional: {contexto_practico}"

    response = openai_client.chat.completions.create(
        model=CONFIG["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"{question}\n\n{context_text}"}
        ],
        temperature=CONFIG["TEMPERATURE"],
        max_tokens=CONFIG["MAX_TOKENS"]
    )

    respuesta = response.choices[0].message.content.strip()
    tokens_usados = response.usage.total_tokens if response.usage else 0
    return respuesta, tokens_usados



# ============= RESPUESTA LEGAL 5 =============
def generate_legal_response5(question, context_docs, contexto_practico=None):
    """
    Firma y retorno idénticos al original:
    - Params: (question, context_docs, contexto_practico=None)
    - Return: (respuesta:str, tokens_usados:int)

    Cambios clave:
    - Compatibilidad GPT-5: usa max_completion_tokens y NO envía temperature.
    - Backward compatible (GPT-3.5/4): usa max_tokens y temperature.
    - Sin documentos: fallback con prefijo obligatorio y orientación general.
    """
    model = CONFIG.get("OPENAI_MODEL", "gpt-5-mini")
    is_gpt5 = str(model).startswith("gpt-5")
    max_out = int(CONFIG.get("MAX_TOKENS", 2000))  # seguimos respetando tu config

    # =============== 1) Sin documentos → Fallback con prefijo obligatorio ===============
    if not context_docs:
        fallback_prefix = "no encontré normativa oficial, sin embargo "
        system_prompt_fb = (
            "Eres un abogado ecuatoriano. No tienes documentos normativos para citar. "
            "Puedes usar conocimiento general para orientar, pero NO cites artículos, códigos ni números de ley; "
            "evita montos y plazos exactos y aclara que es orientación general. "
            f"Tu respuesta DEBE comenzar EXACTAMENTE con: \"{fallback_prefix}\" (respetando minúsculas) "
            "y luego ofrecer orientación breve y útil. Sugiere confirmar en fuentes oficiales (Función Judicial, SRI, IESS, MDT) "
            "y consultar con un abogado cuando aplique."
        )
        user_msg_fb = (
            "Formato:\n"
            f"- Comienza exacto con: \"{fallback_prefix}\".\n"
            "- Ofrece 5–8 frases o bullets prácticos, claros y accionables.\n"
            "- Indica que es orientación general sin base normativa.\n\n"
            f"Pregunta del usuario:\n{question}"
        )

        kwargs = dict(model=model, messages=[
            {"role": "system", "content": system_prompt_fb},
            {"role": "user", "content": user_msg_fb}
        ])
        if is_gpt5:
            kwargs["max_completion_tokens"] = max_out
        else:
            kwargs["temperature"] = CONFIG.get("TEMPERATURE", 0.3)
            kwargs["max_tokens"] = max_out

        resp = openai_client.chat.completions.create(**kwargs)
        respuesta = (resp.choices[0].message.content or "").strip()
        tokens_usados = resp.usage.total_tokens if getattr(resp, "usage", None) else 0
        return respuesta, tokens_usados

    # =============== 2) Con documentos → Modo estricto (solo lo del contexto) ===============
    system_prompt = """
Eres un abogado ecuatoriano. RESPONDE SOLO con base en los “DOCUMENTOS LEGALES” provistos.
Prohibido conocimiento externo, suposiciones o jurisprudencia que no esté en esos documentos.

Estilo:
- Claro y directo, lenguaje llano.
- Solo una frase empática si hay angustia evidente.
- Prioriza los documentos en el orden entregado.

Estructura obligatoria:
1) Respuesta directa (2–4 frases) que resuelva la duda con palabras sencillas.
2) Fundamento con citas: menciona [CÓDIGO Art. N] por cada afirmación.
3) Citas textuales breves (10–30 palabras) de los artículos, entre comillas.
4) Cierre EXACTO: “Me baso en [artículos citados]”.

Reglas de rigor:
- No infieras nada que no esté textual en los documentos.
- Si una afirmación no puede trazarse a un artículo citado, elimínala.
- Cuando un artículo se cita varias veces, escribe su referencia una sola vez en el cierre.
""".strip()

    context_text = "\nDOCUMENTOS LEGALES:\n" + "\n".join(
        f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto'][:600]}"
        for doc in context_docs
    )

    if contexto_practico:
        context_text += f"\n\n🧾 Contexto práctico adicional: {contexto_practico}"

    kwargs = dict(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question}\n\n{context_text}"}
    ])
    if is_gpt5:
        kwargs["max_completion_tokens"] = max_out
    else:
        kwargs["temperature"] = CONFIG.get("TEMPERATURE", 0.3)
        kwargs["max_tokens"] = max_out

    response = openai_client.chat.completions.create(**kwargs)
    respuesta = (response.choices[0].message.content or "").strip()
    tokens_usados = response.usage.total_tokens if getattr(response, "usage", None) else 0
    return respuesta, tokens_usados

# ===============  enpresario ===============

def generate_legal_response_empresario(question, context_docs, contexto_practico=None):
    """
    MISMA FIRMA Y RETORNO:
    - Params: (question, context_docs, contexto_practico=None)
    - Return: (respuesta:str, tokens_usados:int)

    Implementa flujo dinámico:
    1) Section Planner (IA) decide 3–10 secciones útiles para gerencia (siempre incluye primero “🧭 Respuesta ejecutiva”
       y al final “📚 Fuentes consultadas”).
    2) Web CSE (gob.ec) para soporte operativo. Si no hay fuentes, el Writer puede usar conocimiento del modelo.
    3) Answer Writer usa: LEGAL_FACTS (tus artículos) -> WEB -> Conocimiento del LLM (fallback).
       Inserta citas inline [CÓDIGO Art. N] cuando aplique.
    4) Quality Gate: verifica secciones planeadas, “answer-first”, tablas si se prometen, y fuentes (si existen).
    5) Agrega ⚖️ Fundamento legal al final (solo con tus context_docs, con citas textuales cortas).
    6) (Opcional) Anexa “🧩 Notas operativas (no normativo)” si se recibe `contexto_practico`.

    NUEVO:
    - Inserta al final de cada sección (excepto “📚 Fuentes consultadas”) un enlace corto “🔗 <ALIAS>”
      a la mejor fuente .gob.ec utilizada para esa sección.
    - Reemplaza el contenido de “📚 Fuentes consultadas” por una lista HTML <ul> con 3–7 enlaces deduplicados por dominio.
    """
    import os, json, re, html
    from urllib.parse import urlparse
    import requests

    # -------- Config del modelo --------
    model = CONFIG.get("OPENAI_MODEL", "gpt-5-mini")
    is_gpt5 = str(model).startswith("gpt-5")
    max_out = int(CONFIG.get("MAX_TOKENS", 2000))
    temperature = float(CONFIG.get("TEMPERATURE", 0.3))

    # -------- Config Google CSE --------
    GOOGLE_API_KEY = CONFIG.get("GOOGLE_SEARCH_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_CX = CONFIG.get("GOOGLE_CX") or os.getenv("GOOGLE_CX")

    # ===================== Helpers =====================

    def _google_cse(q, n=5):
        """Busca en gob.ec. Devuelve lista [{title,link,snippet}]."""
        items = []
        if not (GOOGLE_API_KEY and GOOGLE_CX):
            return items
        try:
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_CX,
                    "q": q,
                    "siteSearch": "gob.ec",
                    "siteSearchFilter": "i",
                    "num": min(max(n, 1), 10),
                },
                timeout=12,
            )
            data = resp.json()
            for it in (data.get("items") or [])[:n]:
                items.append({
                    "title": (it.get("title") or "").strip(),
                    "link": (it.get("link") or "").strip(),
                    "snippet": (it.get("snippet") or "").strip()
                })
        except Exception:
            pass
        return items

    def _mk_snippets(results, cap=8):
        """Lista única por URL (máx cap)."""
        out, seen = [], set()
        for r in results:
            u = r.get("link") or ""
            if u and u not in seen:
                out.append(r)
                seen.add(u)
            if len(out) >= cap:
                break
        return out

    def _mk_web_text(snips):
        """Bloque compacto de fuentes para el prompt del Writer."""
        if not snips:
            return "No se hallaron fuentes oficiales en gob.ec para esta consulta."
        lines = []
        for r in snips:
            t = r.get("title", "Fuente")
            u = r.get("link", "")
            s = r.get("snippet", "")
            lines.append("Título: " + t + "\nURL: " + u + "\nExtracto: " + s)
        return "\n\n".join(lines)

    def _short_quote(txt, min_w=10, max_w=25):
        """Cita corta 10–25 palabras."""
        if not txt:
            return ""
        clean = " ".join(txt.split())
        words = clean.split(" ")
        if len(words) <= max_w:
            return clean
        end = max(min_w, min(max_w, 20))
        return " ".join(words[:end])

    def _mk_fundamento(docs):
        """Bloque final ⚖️ Fundamento legal solo con tus context_docs."""
        articulos, usados = [], []
        for d in docs or []:
            codigo = (d.get("codigo") or "").strip()
            art = (d.get("articulo") or "").strip()
            texto = (d.get("texto") or "").strip()
            if not (codigo and art and texto):
                continue
            ref = f"{codigo} Art. {art}"
            cita = _short_quote(texto, 10, 25)
            articulos.append(f"- [{ref}] “{html.escape(cita)}”.")
            usados.append(ref)
            if len(articulos) >= 6:
                break
        if not articulos:
            return ""
        cierre = "Me baso en [" + ", ".join(usados) + "]."
        return "⚖️ Fundamento legal\n" + "\n".join(articulos) + "\n\n" + cierre

    def _allowed_refs_from_docs(docs):
        """Lista de referencias válidas 'CÓDIGO Art. N' para validar citas inline (si lo usas)."""
        refs = []
        for d in docs or []:
            codigo = (d.get("codigo") or "").strip()
            art = (d.get("articulo") or "").strip()
            if codigo and art:
                refs.append(f"{codigo} Art. {art}")
        return set(refs)

    def _compact_legal_text(docs, cap=12):
        """Bloque compacto de artículos para el Writer (no URLs)."""
        lines = []
        for d in (docs or [])[:cap]:
            cod = (d.get("codigo") or "").strip()
            art = (d.get("articulo") or "").strip()
            txt = (d.get("texto") or "").strip()
            if cod and art and txt:
                lines.append(f"{cod} Art.{art}: {txt[:600]}")
        return "\n".join(lines) if lines else "(sin documentos)"

    def _answer_first_ok(txt):
        """Valida que la primera viñeta de 🧭 Respuesta ejecutiva tome postura clara."""
        m = re.search(r"🧭\s*Respuesta\s+ejecutiva\s*(.+?)(?:\n\n[^\n]|$)", txt, flags=re.IGNORECASE | re.DOTALL)
        sec = m.group(1) if m else ""
        fb = re.search(r"^\s*[-•]\s*(.+)$", sec, flags=re.MULTILINE)
        if not fb:
            return False
        bullet = fb.group(1).strip().lower()
        keywords = ["sí", "si,", "no", "depende", "es legal", "no es legal", "permitido", "prohibido", "puedes", "no puedes"]
        return any(k in bullet for k in keywords)

    def _has_markdown_table(txt):
        """Chequea si hay una tabla Markdown básica en el texto."""
        return bool(re.search(r"^\|.+\|\s*\n\|[-:\s|]+\|\s*\n(\|.*\|\s*\n)+", txt, flags=re.MULTILINE))

    # ---------- NUEVOS helpers para enlaces por sección ----------

    STOP_ES = {"de","la","las","los","un","una","uno","y","o","u","para","por","con","sin","del","al","en","que","como","cuando","donde","cuál","cuáles","segun","ante","entre","hacia","hasta","sobre","tras","este","esta","esto"}

    def _domain_alias(url: str) -> str:
        """Devuelve alias corto para dominios gob.ec (SRI, IESS, MDT, SENADI…)."""
        try:
            d = urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return "FUENTE"
        mapping = {
            "sri.gob.ec": "SRI",
            "iess.gob.ec": "IESS",
            "trabajo.gob.ec": "MDT",
            "ministeriodeltrabajo.gob.ec": "MDT",
            "derechosintelectuales.gob.ec": "SENADI",
            "snai.gob.ec": "SNAI",
            "uafe.gob.ec": "UAFE",
            "sercop.gob.ec": "SERCOP",
            "registrocivil.gob.ec": "REGISTRO CIVIL",
            "funcionjudicial.gob.ec": "FUNCIÓN JUDICIAL",
            "corteconstitucional.gob.ec": "CORTE CONSTITUCIONAL",
            "ant.gob.ec": "ANT",
            "arcotel.gob.ec": "ARCOTEL",
            "arcsa.gob.ec": "ARCSA",
            "senescyt.gob.ec": "SENESCYT",
            "aduana.gob.ec": "SENAE",
        }
        if d in mapping:
            return mapping[d]
        # Si es .gob.ec: usa subdominio como alias (ej. "midena.gob.ec" -> "MIDENA")
        return (d.split(".")[0].upper() if d.endswith(".gob.ec") else d.upper())

    def _norm_terms(text: str) -> set:
        t = re.sub(r"[^a-záéíóúñü0-9 ]+", " ", (text or "").lower())
        return {w for w in t.split() if len(w) > 2 and w not in STOP_ES}

    def _best_snippet_for(section_text: str, snips: list, min_overlap: int = 2):
        """Escoge el snippet web más afín a la sección por solapamiento de términos."""
        st = _norm_terms(section_text)
        best, score = None, 0
        for sn in snips or []:
            mix = " ".join([sn.get("title",""), sn.get("snippet","")])
            sc = len(st & _norm_terms(mix))
            if sc > score:
                best, score = sn, sc
        return best if score >= min_overlap else None

    def _split_by_sections(respuesta: str, sections: list):
        """Encuentra offsets de cada sección por el título exacto (línea completa)."""
        hits = []
        for s in sections:
            m = re.search(r"^" + re.escape(s) + r"\s*$", respuesta, flags=re.MULTILINE)
            if m: hits.append((s, m.start()))
        hits.sort(key=lambda x: x[1])
        chunks = []
        for i, (s, start) in enumerate(hits):
            end = hits[i+1][1] if i+1 < len(hits) else len(respuesta)
            chunks.append((s, start, end))
        return chunks

    def _attach_links_at_section_ends(respuesta: str, sections: list, snips: list) -> str:
        """Añade '🔗 <alias>' con enlace al final de cada sección (excepto Fuentes)."""
        chunks = _split_by_sections(respuesta, sections)
        if not chunks:
            return respuesta
        used = set()
        new_parts, last_end = [], 0
        for (title, start, end) in chunks:
            # Copia lo previo sin tocar
            new_parts.append(respuesta[last_end:start])
            segment = respuesta[start:end]
            if title.strip() != "📚 Fuentes consultadas":
                # Evitar reutilizar el mismo link en muchas secciones
                remaining = [r for r in (snips or []) if (r.get("link") or "") not in used]
                cand = _best_snippet_for(segment, remaining)
                if cand and cand.get("link"):
                    alias = _domain_alias(cand["link"])
                    link_html = f'\n\n🔗 <a href="{html.escape(cand["link"])}" target="_blank" rel="noopener nofollow">{html.escape(alias)}</a>'
                    segment = segment.rstrip() + link_html + "\n"
                    used.add(cand["link"])
            new_parts.append(segment)
            last_end = end
        new_parts.append(respuesta[last_end:])
        return "".join(new_parts)

    def _mk_fuentes_html(snips, cap=7):
        """Lista HTML de fuentes (deduplicadas por dominio)."""
        if not snips:
            return "<p>—</p>"
        used_domains, items = set(), []
        for r in snips:
            u = (r.get("link") or "").strip()
            if not u:
                continue
            d = urlparse(u).netloc.replace("www.", "")
            if d in used_domains:
                continue
            t = html.escape(r.get("title") or d)
            items.append(f'<li><a href="{html.escape(u)}" target="_blank" rel="noopener nofollow">{t}</a> — {html.escape(d)}</li>')
            used_domains.add(d)
            if len(items) >= cap:
                break
        return "<ul>" + "\n".join(items) + "</ul>"

    # ===================== Inicio del flujo =====================
    tokens_total = 0
    q = (question or "").strip()

    # ---------- A) SECTION PLANNER (elige 3–10 secciones) ----------
    planner_system = (
        "Eres un planificador de respuesta para un gerente en Ecuador. "
        "Elige ENTRE 3 Y 10 secciones con títulos claros (máx 35 caracteres cada uno) que mejor respondan la pregunta. "
        "SIEMPRE incluye como primera sección '🧭 Respuesta ejecutiva' y como última '📚 Fuentes consultadas'. "
        "Si el tema requiere plazos/responsables, incluye una sección con tabla. "
        "Devuelve SOLO JSON válido con: {\"sections\": [\"título1\", ...]} "
        "(no agregues comentarios)."
    )
    planner_user = 'Pregunta: """' + q + '"""'

    kwargs_pl = dict(model=model, messages=[
        {"role": "system", "content": planner_system},
        {"role": "user", "content": planner_user}
    ])
    if is_gpt5:
        kwargs_pl["max_completion_tokens"] = 180
    else:
        kwargs_pl["temperature"] = max(0.2, temperature - 0.1)
        kwargs_pl["max_tokens"] = 180

    try:
        pl = openai_client.chat.completions.create(**kwargs_pl)
        plan_txt = (pl.choices[0].message.content or "").strip()
        tokens_total += pl.usage.total_tokens if getattr(pl, "usage", None) else 0
        try:
            plan = json.loads(plan_txt)
            sections = [s.strip() for s in (plan.get("sections") or []) if isinstance(s, str) and s.strip()]
        except Exception:
            sections = []
    except Exception:
        sections = []

    # Salvaguardas: garantizar cabeceras mínimas
    if not sections:
        sections = ["🧭 Respuesta ejecutiva", "🗓️ Plazos y responsables", "💸 Multas y riesgos", "✅ Acciones inmediatas", "🧾 Checklist", "❌ Errores comunes", "📚 Fuentes consultadas"]
    # Normalizar y recortar 3–10
    if sections[0] != "🧭 Respuesta ejecutiva":
        sections = ["🧭 Respuesta ejecutiva"] + [s for s in sections if s != "🧭 Respuesta ejecutiva"]
    if "📚 Fuentes consultadas" not in sections:
        sections.append("📚 Fuentes consultadas")
    if len(sections) < 3:
        sections = sections + ["✅ Acciones inmediatas"]
    sections = sections[:10]
    if sections[-1] != "📚 Fuentes consultadas":
        sections = [s for s in sections if s != "📚 Fuentes consultadas"]
        sections.append("📚 Fuentes consultadas")

    # Detecta si hay una sección que implique tabla (por nombre)
    wants_table = any(any(k in s.lower() for k in ["plazo", "tabla", "responsable"]) for s in sections)

    # ---------- B) FUENTES: LEGAL_FACTS + WEB (gob.ec) ----------
    legal_facts_text = _compact_legal_text(context_docs)
    allowed_refs = _allowed_refs_from_docs(context_docs)  # (por si validas las citas inline)

    # WEB: Queries base (planner no retorna queries; usamos heurística + pregunta)
    queries = [
        q,
        q + " site:gob.ec",
        q + " site:trabajo.gob.ec",
        q + " site:iess.gob.ec",
        q + " site:sri.gob.ec",
        q + " site:derechosintelectuales.gob.ec",
    ]
    # Heurística temática simple
    ql = q.lower()
    if any(p in ql for p in ["contrato", "despido", "jornada", "sut", "ministerio", "trabajo"]):
        queries.append(q + " procedimiento site:trabajo.gob.ec")
    if any(p in ql for p in ["iess", "afili", "aporte"]):
        queries.append(q + " site:iess.gob.ec")
    if any(p in ql for p in ["iva", "retención", "rdep", "ats", "sri", "formulario"]):
        queries.append(q + " site:sri.gob.ec")

    all_results = []
    for search_q in queries[:6]:
        all_results.extend(_google_cse(search_q, n=4))
    web_snippets = _mk_snippets(all_results, cap=8)
    web_context_text = _mk_web_text(web_snippets)

    # ---------- C) ANSWER WRITER ----------
    writer_system = (
        "Eres un abogado corporativo ecuatoriano para gerentes. "
        "Responde ‘answer-first’ y llena TODAS las secciones listadas por el plan. "
        "Prioridad de información: 1) LEGAL_FACTS (tus artículos). 2) WEB_SNIPPETS (gob.ec). 3) Conocimiento propio si no hay fuentes. "
        "Si aplicas una regla de LEGAL_FACTS, cierra la oración con [CÓDIGO Art. N]. "
        "Si falta un dato, usa “—” en vez de inventar. "
        "No coloques enlaces en el cuerpo; las referencias web se añadirán automáticamente al final de cada sección. "
        "En '📚 Fuentes consultadas', lista 3–7 títulos + dominio (sin repetir dominios). "
        "Si no hay ninguna fuente web válida, igualmente escribe la respuesta basada en conocimiento general y coloca '—' en Fuentes."
    )
    writer_user = (
        "PREGUNTA:\n" + q +
        "\n\nSECCIONES_PLAN (usa EXACTAMENTE estos encabezados, en orden):\n" + "\n".join(["- " + s for s in sections]) +
        "\n\nLEGAL_FACTS (usa como primera prioridad y cita inline [CÓDIGO Art. N] cuando corresponda):\n" + legal_facts_text +
        "\n\nWEB_SNIPPETS (solo gob.ec; si está vacío, puedes usar conocimiento del modelo):\n" + web_context_text +
        ("\n\nREQUISITO_TABLA: sí" if wants_table else "\n\nREQUISITO_TABLA: no") +
        "\n\nREGLAS DURAS:\n- No inventes montos/plazos.\n- Usa “—” si un dato no aparece.\n- Tono profesional, directo, orientado a decisión.\n"
    )

    kwargs_w = dict(model=model, messages=[
        {"role": "system", "content": writer_system},
        {"role": "user", "content": writer_user}
    ])
    if is_gpt5:
        kwargs_w["max_completion_tokens"] = max_out
    else:
        kwargs_w["temperature"] = temperature
        kwargs_w["max_tokens"] = max_out

    wr = openai_client.chat.completions.create(**kwargs_w)
    respuesta = (wr.choices[0].message.content or "").strip()
    tokens_total += wr.usage.total_tokens if getattr(wr, "usage", None) else 0

    # ---------- D) QUALITY GATE ----------
    # 1) Todas las secciones del plan presentes (en orden flexible pero con todos los títulos exactos)
    missing = []
    for s in sections:
        if re.search(r"^" + re.escape(s) + r"\s*$", respuesta, flags=re.IGNORECASE | re.MULTILINE) is None:
            missing.append(s)

    # 2) Answer-first en la primera viñeta
    needs_answer_first = not _answer_first_ok(respuesta)

    # 3) Si se prometió tabla, verificar que exista al menos una tabla Markdown
    needs_table = wants_table and (not _has_markdown_table(respuesta))

    if missing or needs_answer_first or needs_table:
        problems = []
        if missing: problems.append("Faltan secciones: " + ", ".join(missing))
        if needs_answer_first: problems.append("La primera viñeta de ‘🧭 Respuesta ejecutiva’ no toma postura clara.")
        if needs_table: problems.append("Se prometió tabla de plazos/responsables pero no se encontró una tabla.")

        fix_system = (
            "Corrige la respuesta para cumplir EXACTAMENTE con el plan de secciones y reglas. "
            "Mantén el contenido ya correcto; añade o ajusta solo lo necesario. "
            "Si prometiste tabla de plazos/responsables, incluye una tabla Markdown. "
            "La primera viñeta de '🧭 Respuesta ejecutiva' debe tomar postura clara (‘Sí, con condiciones…’, ‘No, porque…’, ‘Depende: si… entonces…’). "
            "No inventes datos: usa “—” si falta."
        )
        fix_user = (
            "; ".join(problems) +
            "\n\nSECCIONES_PLAN:\n" + "\n".join(["- " + s for s in sections]) +
            ("\n\nREQUISITO_TABLA: sí" if wants_table else "\n\nREQUISITO_TABLA: no") +
            "\n\nRespuesta actual:\n" + respuesta
        )
        kwargs_fix = dict(model=model, messages=[
            {"role": "system", "content": fix_system},
            {"role": "user", "content": fix_user}
        ])
        if is_gpt5:
            kwargs_fix["max_completion_tokens"] = max_out
        else:
            kwargs_fix["temperature"] = max(0.2, temperature - 0.1)
            kwargs_fix["max_tokens"] = max_out

        wr2 = openai_client.chat.completions.create(**kwargs_fix)
        fixed = (wr2.choices[0].message.content or "").strip()
        if fixed:
            respuesta = fixed
            tokens_total += wr2.usage.total_tokens if getattr(wr2, "usage", None) else 0

    # ---------- E) Enlaces por sección + Fuentes HTML ----------
    # Inserta "🔗 <ALIAS>" al final de cada sección (excepto la de fuentes)
    respuesta = _attach_links_at_section_ends(respuesta, sections, web_snippets)

    # Reemplaza el contenido de “📚 Fuentes consultadas” con lista HTML
    fuentes_html = _mk_fuentes_html(web_snippets, cap=7)
    new_respuesta = re.sub(r"(📚\s*Fuentes consultadas\s*)[\s\S]*$", r"\1\n" + fuentes_html, respuesta)
    if new_respuesta == respuesta:
        # Por si acaso no encontró la sección (no debería pasar), la agregamos al final
        respuesta = respuesta.rstrip() + "\n\n📚 Fuentes consultadas\n" + fuentes_html
    else:
        respuesta = new_respuesta

    # ---------- F) Notas operativas (opcional) ----------
    if contexto_practico:
        respuesta += "\n\n🧩 Notas operativas (no normativo)\n" + (contexto_practico.strip()[:1200])

    # ---------- G) Fundamento legal (tus context_docs) ----------
    if context_docs:
        bloque = _mk_fundamento(context_docs)
        if bloque:
            # si el Writer hubiera agregado uno propio (no debería), lo removemos y ponemos el nuestro
            if "⚖️ Fundamento legal" in respuesta:
                respuesta = re.split(r"\n?⚖️ Fundamento legal.*", respuesta, maxsplit=1)[0].rstrip()
            respuesta += "\n\n" + bloque

    return respuesta, tokens_total


# ============= RESPUESTA EMPRESARIO API =============
def generate_legal_response_empresario_API(question, context_docs, contexto_practico=None):
    """
    Versión estilo Playground:
    - Usa Responses API con la herramienta web_search (GPT-5).
    - No usa context_docs ni añade ⚖️ Fundamento legal.
    - Formato práctico con sección de “📚 Fuentes consultadas” (<ul>).
    """
    import os, re, html, logging
    from urllib.parse import urlparse

    # -------- Config del modelo (forzamos GPT-5; fallback a 5-mini si hace falta) --------
    _cfg = globals().get("CONFIG", {}) if "CONFIG" in globals() else {}
    model_preferido = _cfg.get("OPENAI_MODEL_RESPONSES") or "gpt-5"
    if not str(model_preferido).startswith("gpt-5"):
        model_preferido = "gpt-5"
    model_fallback = "gpt-5-mini"

    # Límite de salida razonable para mantener costo bajo en esta ruta
    max_out = int(_cfg.get("MAX_TOKENS", 1800))
    max_out = max(256, min(max_out, 1800))  # clamp defensivo

    # -------- Cliente OpenAI (usa OPENAI_PROJECT si existe; requerido por tools en algunas cuentas) --------
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        project = os.getenv("OPENAI_PROJECT")
        if not api_key:
            return ("⚠️ Falta OPENAI_API_KEY en el entorno.", 0)
        client = globals().get("openai_client", None)
        if client is None or not hasattr(client, "responses"):
            client = OpenAI(api_key=api_key, project=project) if project else OpenAI(api_key=api_key)
    except Exception as e:
        return ("⚠️ Falta SDK OpenAI o error creando cliente: " + str(e), 0)

    # -------- Helpers --------
    def _safe_output_text(resp):
        """Devuelve texto aunque output_text venga vacío."""
        txt = (getattr(resp, "output_text", "") or "").strip()
        if txt:
            return txt
        parts = []
        for item in (getattr(resp, "output", []) or []):
            try:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        t = getattr(c, "text", None)
                        if t:
                            parts.append(t)
            except Exception:
                pass
        return "\n".join(p for p in parts if p).strip()

    def _mk_fuentes_html_from_urls(urls, titles=None, cap=7):
        titles = titles or {}
        used_domains, items = set(), []
        for u in urls:
            if not u:
                continue
            try:
                d = urlparse(u).netloc.replace("www.", "")
            except Exception:
                d = ""
            if d in used_domains:
                continue
            t = html.escape(titles.get(u, d or "Fuente"))
            items.append(f'<li><a href="{html.escape(u)}" target="_blank" rel="noopener nofollow">{t}</a> — {html.escape(d or "")}</li>')
            used_domains.add(d)
            if len(items) >= cap:
                break
        return "<ul>" + "\n".join(items) + "</ul>" if items else "<p>—</p>"

    def _extract_citations(resp):
        """Recolecta URLs/títulos desde items/citations y, si falta, por regex del texto."""
        urls, titles = [], {}
        # 1) Intentar desde metadatos de items
        for item in (getattr(resp, "output", []) or []):
            try:
                meta = {}
                if isinstance(item, dict):
                    meta = (item.get("citation") or item.get("metadata") or item.get("source") or {})
                else:
                    meta = getattr(item, "citation", None) or getattr(item, "metadata", None) or {}
                url = (meta.get("url") or meta.get("href") or meta.get("source_url") or "").strip()
                ttl = (meta.get("title") or meta.get("source_title") or "").strip()
                if url and url not in urls:
                    urls.append(url)
                    if ttl:
                        titles[url] = ttl
            except Exception:
                pass
        # 2) Regex del texto si no hubo nada
        text = (getattr(resp, "output_text", "") or "")
        if not urls and text:
            for m in re.findall(r"https?://[^\s\)\]\}<>\"']+", text):
                if m not in urls:
                    urls.append(m)
        return urls, titles

    # -------- Prompt estilo Playground --------
    q = (question or "").strip()

    SYSTEM = (
        "Eres un asesor experto en trámites empresariales en Ecuador (es-EC, tz: America/Guayaquil). "
        "Usa la herramienta Web Search de forma ACTIVA priorizando dominios oficiales "
        "(supercias.gob.ec, sri.gob.ec, gob.ec, funcionjudicial.gob.ec, registros mercantiles). "
        "Responde answer-first y NO inventes montos/plazos: si no están claros, escribe “—” y explica cómo obtenerlos. "
        "FORMATO (máx 8 secciones):\n"
        "1) 🧭 Respuesta ejecutiva (bullets, veredicto claro)\n"
        "2) Pasos oficiales\n"
        "3) Documentos mínimos\n"
        "4) Costos y tasas (si hay)\n"
        "5) Plazos típicos (si hay)\n"
        "6) ✅ Acciones inmediatas / ❌ Errores comunes (opcional)\n"
        "7) 🧩 Tips operativos (opcional)\n"
        "8) 📚 Fuentes consultadas (obligatorio) como <ul> 5–7 enlaces (título + dominio), sin repetir dominio.\n"
        "No pegues textos extensos de fuentes; sé conciso y útil."
    )

    USER = (
        "OBJETIVO: igualar el estilo del Playground con pasos oficiales, documentos mínimos, costos/tasas y plazos, con enlaces oficiales.\n\n"
        "PREGUNTA:\n" + q
    )

    # -------- Llamada principal (web_search) --------
    def _call_with_model(_model, use_preview=False):
        tool = {"type": "web_search_preview"} if use_preview else {"type": "web_search"}
        return client.responses.create(
            model=_model,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": USER}],
            tools=[tool],
            max_output_tokens=max_out,   # >= 16
            # Importante: NO enviar temperature con modelos gpt-5
        )

    tokens_total = 0
    respuesta = ""
    used_model = model_preferido
    try:
        r = _call_with_model(used_model, use_preview=False)
    except Exception:
        # Intento alterno con web_search_preview
        try:
            r = _call_with_model(used_model, use_preview=True)
        except Exception:
            # Fallback a gpt-5-mini con web_search
            try:
                used_model = model_fallback
                r = _call_with_model(used_model, use_preview=False)
            except Exception:
                # Último intento: gpt-5-mini con preview
                r = _call_with_model(used_model, use_preview=True)

    # Texto y uso
    respuesta = _safe_output_text(r)
    try:
        u = getattr(r, "usage", None)
        if u:
            tokens_total = (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0)
    except Exception:
        pass

    # Reintento sin tools si no hubo texto (síntesis mínima)
    if not respuesta.strip():
        r2 = client.responses.create(
            model=used_model,
            input=[
                {"role": "system", "content": "Responde en el formato indicado, AUN SIN BUSCAR. Usa “—” si faltan datos."},
                {"role": "user", "content": USER},
            ],
            max_output_tokens=min(max_out, 800),
        )
        respuesta = _safe_output_text(r2)
        try:
            u2 = getattr(r2, "usage", None)
            if u2:
                tokens_total += (getattr(u2, "input_tokens", 0) or 0) + (getattr(u2, "output_tokens", 0) or 0)
        except Exception:
            pass

    # -------- Forzar sección de Fuentes consultadas (HTML <ul>) --------
    # Si falta, o no hay <ul>, construimos una con URLs detectadas
    urls, titles = _extract_citations(r)
    fuentes_html = _mk_fuentes_html_from_urls(urls, titles, cap=7)

    if "📚 Fuentes consultadas" not in respuesta:
        respuesta = respuesta.rstrip() + "\n\n📚 Fuentes consultadas\n" + fuentes_html
    else:
        if "<ul>" not in respuesta:
            # Reemplazar bloque final de fuentes por nuestra lista HTML
            respuesta = re.sub(r"(📚\s*Fuentes consultadas\s*)[\s\S]*$",
                               r"\\1\n" + fuentes_html, respuesta)

    return respuesta, tokens_total






# ============= RESPUESTA PRÁCTICA =============

def obtener_respuesta_practica(question, score=None):
    practical_index_name = "indice-respuestas-abogados"
    practical_index = pc.Index(practical_index_name)

    practical_vector_store = PineconeVectorStore(
        pinecone_index=practical_index
    )

    practical_index_instance = VectorStoreIndex.from_vector_store(
        vector_store=practical_vector_store,
        service_context=service_context
    )

    engine = practical_index_instance.as_query_engine(similarity_top_k=1)
    resultado = engine.query(question)

    if not resultado.source_nodes:
        return None

    texto_practico = resultado.response.strip()

    # Elegir introducción según el score
    if score is not None and score < 0.75:
        introduccion = (
            "❗ La respuesta no responde directamente a la pregunta del usuario.\n"
            "- Introduce con una frase como:\n"
            "  \"Es difícil indicarte si [reformula aquí la intención del usuario], en este caso es importante que un abogado experto te asesore. Sin embargo, puedo decirte que...\"\n"
            "- Reformula después el contenido original como referencia general.\n"
            "- No afirmes nada que no esté expresamente en el texto original.\n"
        )
    else:
        introduccion = (
            "✅ La respuesta es clara y útil para la pregunta del usuario:\n"
            "- Reformúlala sin alterar el mensaje, con un tono claro, amable y profesional.\n"
        )

    # Prompt final
    prompt = (
        "Reformula esta respuesta práctica legal para que suene humana, empática, cercana y útil para alguien sin conocimientos jurídicos. Usa segunda persona. Evalúa si responde o no directamente a la siguiente pregunta:\n\n"
        f"🧑‍⚖️ Pregunta del usuario: \"{question}\"\n\n"
        f"{introduccion}\n"
        "🔒 Reglas adicionales:\n"
        "- Conserva enlaces web útiles como http://consultas.funcionjudicial.gob.ec si están presentes en el texto original.\n"
        "- NO agregues enlaces si no están.\n"
        "- Elimina nombres propios, montos específicos, fechas y datos sensibles.\n\n"
        f"Texto original:\n{texto_practico}"
    )

    # Llamada a OpenAI
    reformulado = openai_client.chat.completions.create(
        model=CONFIG["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": "Eres un asistente legal empático que habla en tono claro y humano."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=800
    )

    return reformulado.choices[0].message.content.strip()


# ============= ENDPOINT PRINCIPAL =============
@app.route("/query", methods=["GET", "POST"])
def handle_query():
    try:
        question = request.args.get("question", "").strip() if request.method == "GET" else request.get_json().get("question", "").strip()
        if not question:
            return jsonify({"error": "Se requiere 'question'"}), 400

        # ========== CONTEXTO LEGAL ==========
        query_engine = index.as_query_engine(similarity_top_k=TOP_K_RESULTS)
        pinecone_response = query_engine.query(question)

        context_docs = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}

        total_docs = len(pinecone_response.source_nodes)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        for i, node in enumerate(pinecone_response.source_nodes):
            metadata = getattr(node.node, 'metadata', {})
            codigo = metadata.get('code', '')
            articulo = metadata.get('article', '')
            texto = getattr(node.node, 'text', '') or metadata.get("text", '')
            texto = texto.strip()

            doc_data = {"codigo": codigo, "articulo": articulo, "texto": texto}
            context_docs.append(doc_data)

            if i < alta_limite:
                biografia_juridica["alta"].append(doc_data)
            elif i < media_limite:
                biografia_juridica["media"].append(doc_data)
            else:
                biografia_juridica["baja"].append(doc_data)

        if not context_docs:
            return jsonify({"respuesta": "No encontré normativa aplicable. No me baso en ningún artículo."})

        # ========== RESPUESTAS ==========
        respuesta_legal, tokens_usados = generate_legal_response5(question, context_docs)

        index_respuestas_abogados = pc.Index("indice-respuestas-abogados")
        embedding = embed_model._get_query_embedding(question)
        similares = index_respuestas_abogados.query(vector=embedding, top_k=1, include_metadata=True)

        respuesta_practica_reformulada = None

        if similares.get("matches"):
            match = similares["matches"][0]
            score = match.get("score", 0)
            if score >= 0.6:
                respuesta_practica_reformulada = obtener_respuesta_practica(question, score=score)
            else:
                respuesta_practica_reformulada = None

        # ========== UNIFICAR RESPUESTA ==========
        bloques = []

        if respuesta_practica_reformulada:
            bloques.append("📌 Recomendación práctica:\n" + respuesta_practica_reformulada.strip())

        bloques.append("⚖️ Fundamento legal:\n" + respuesta_legal.strip())

        return jsonify({
            "respuesta": "\n\n---\n\n".join(bloques),
            "biografia_juridica": biografia_juridica,
            "tokens_usados": {"total_tokens": tokens_usados}
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500



# ============= ENDPOINT PRINCIPAL =============
@app.route("/queryEmpresario", methods=["GET", "POST"])
def handle_query_empresario():
    try:
        question = request.args.get("question", "").strip() if request.method == "GET" else request.get_json().get("question", "").strip()
        if not question:
            return jsonify({"error": "Se requiere 'question'"}), 400

        # ========== CONTEXTO LEGAL ==========
        query_engine = index.as_query_engine(similarity_top_k=TOP_K_RESULTS)
        pinecone_response = query_engine.query(question)

        context_docs = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}

        total_docs = len(pinecone_response.source_nodes)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        for i, node in enumerate(pinecone_response.source_nodes):
            metadata = getattr(node.node, 'metadata', {})
            codigo = metadata.get('code', '')
            articulo = metadata.get('article', '')
            texto = getattr(node.node, 'text', '') or metadata.get("text", '')
            texto = texto.strip()

            doc_data = {"codigo": codigo, "articulo": articulo, "texto": texto}
            context_docs.append(doc_data)

            if i < alta_limite:
                biografia_juridica["alta"].append(doc_data)
            elif i < media_limite:
                biografia_juridica["media"].append(doc_data)
            else:
                biografia_juridica["baja"].append(doc_data)

      
        # ========== RESPUESTAS ==========
        respuesta_legal, tokens_usados = generate_legal_response_empresario_API(question, context_docs)

        index_respuestas_abogados = pc.Index("indice-respuestas-abogados")
        embedding = embed_model._get_query_embedding(question)
        similares = index_respuestas_abogados.query(vector=embedding, top_k=1, include_metadata=True)

        respuesta_practica_reformulada = None

        if similares.get("matches"):
            match = similares["matches"][0]
            score = match.get("score", 0)
            if score >= 0.6:
                respuesta_practica_reformulada = obtener_respuesta_practica(question, score=score)
            else:
                respuesta_practica_reformulada = None

        # ========== UNIFICAR RESPUESTA ==========
        bloques = []

        if respuesta_practica_reformulada:
            bloques.append("📌 Recomendación práctica:\n" + respuesta_practica_reformulada.strip())

        bloques.append("⚖️ Fundamento legal:\n" + respuesta_legal.strip())

        return jsonify({
            "respuesta": "\n\n---\n\n".join(bloques),
            "biografia_juridica": biografia_juridica,
            "tokens_usados": {"total_tokens": tokens_usados}
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

import tiktoken

from flask import send_from_directory
from docx import Document
import uuid
import os

# Endpoint de descarga (añádelo solo una vez en tu app)
@app.route("/descargar/<filename>")
def descargar_archivo(filename):
    return send_from_directory("archivos_temp", filename, as_attachment=True)

# Función auxiliar para guardar y generar link
def guardar_docx_y_retornar_link(texto, host_url):
    filename = f"{uuid.uuid4()}.docx"
    filepath = os.path.join("archivos_temp", filename)
    os.makedirs("archivos_temp", exist_ok=True)

    doc = Document()
    for linea in texto.split('\n'):
        doc.add_paragraph(linea)
    doc.save(filepath)

    return f"{host_url}descargar/{filename}"

# ======================= GENERAR CONTRATO COMPLETO =======================
@app.route("/generar-contrato-completo", methods=["POST"])
def generar_contrato_completo():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()

        if not pregunta:
            return jsonify({"error": "La pregunta es obligatoria"}), 400

        contrato_query_engine = contratos_index.as_query_engine(similarity_top_k=1)
        resultado_contrato = contrato_query_engine.query(pregunta)

        contrato_base = ""
        if resultado_contrato.source_nodes:
            contrato_base = resultado_contrato.source_nodes[0].node.text.strip()

        if contrato_base:
            prompt = f"""
Eres un abogado ecuatoriano experto en redacción de documentos legales. A continuación tienes un modelo jurídico que debes adaptar para responder a la solicitud del usuario. Mantén su estructura y estilo, pero personaliza el contenido según la petición.

📄 Solicitud del usuario:
{pregunta}

📑 Modelo de referencia:
{contrato_base}

✍️ Instrucciones:
- No incluyas explicaciones, solo el documento.
- Usa lenguaje jurídico claro.
- Usa campos genéricos como [NOMBRE], [FECHA], etc.
""".strip()

            def contar_tokens(texto):
                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                return len(enc.encode(texto))

            tokens_prompt = contar_tokens(prompt)

            if tokens_prompt <= 3000:
                modelo = "gpt-3.5-turbo"
                max_tokens_salida = 1000
            else:
                modelo = "gpt-4o"
                max_tokens_salida = 8000

            response = openai_client.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": "Eres un abogado ecuatoriano experto en redacción legal."},
                    {"role": "user", "content": prompt}
                ],
                temperature=CONFIG["TEMPERATURE"],
                max_tokens=max_tokens_salida
            )

            texto = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens

            if tokens > 8000:
                url = guardar_docx_y_retornar_link(texto, request.host_url)
                return jsonify({
                    "respuesta": f"✅ El documento ha sido generado correctamente. <a href='{url}' target='_blank'>Haz clic aquí para descargarlo en Word (.docx)</a>.",
                    "tokens_usados": { "total_tokens": tokens },
                    "biografia_juridica": None
                })

            return jsonify({
                "respuesta": texto,
                "tokens_usados": { "total_tokens": tokens },
                "biografia_juridica": None
            })

        query_engine = index.as_query_engine(similarity_top_k=10)
        resultado = query_engine.query(pregunta)

        contexto_legal = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}

        total_docs = len(resultado.source_nodes)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        for i, nodo in enumerate(resultado.source_nodes):
            meta = getattr(nodo.node, 'metadata', {})
            codigo = meta.get("code", "")
            articulo = meta.get("article", "")
            texto = getattr(nodo.node, 'text', '') or meta.get("text", '')
            texto = texto.strip()

            if codigo and articulo:
                contexto_legal.append(f"{codigo} Art. {articulo}: {texto[:500]}")
                doc = {"codigo": codigo, "articulo": articulo, "texto": texto}

                if i < alta_limite:
                    biografia_juridica["alta"].append(doc)
                elif i < media_limite:
                    biografia_juridica["media"].append(doc)
                else:
                    biografia_juridica["baja"].append(doc)

        prompt = f"""
Eres un abogado ecuatoriano experto en redacción de documentos legales. Vas a redactar un texto profesional, completo y jurídicamente válido, en respuesta a la solicitud del usuario.

✍️ Instrucciones estrictas:
- Redacta directamente el documento legal solicitado, sin explicaciones ni introducciones.
- Usa lenguaje legal claro y preciso, adecuado al sistema jurídico del Ecuador.
- Si el documento requiere estructura (contrato, demanda, reglamento, etc.), incluye numeración adecuada: cláusulas, artículos, incisos.
- Si el documento es breve (como una solicitud o escrito procesal), redacta en formato carta legal.
- Utiliza campos genéricos para datos personales: [NOMBRE], [FECHA], [CANTIDAD], [CIUDAD], etc.
- No agregues artículos legales inventados: usa solo los del contexto.

📄 Solicitud del usuario:
{pregunta}

📚 Contexto legal (solo puedes usar esto):
{chr(10).join(contexto_legal)}

🧾 Al final del documento, incluye una sección con el encabezado “Fundamento legal” y menciona los artículos usados.
""".strip()

        response = openai_client.chat.completions.create(
            model=CONFIG["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": "Eres un abogado ecuatoriano experto en redacción legal."},
                {"role": "user", "content": prompt}
            ],
            temperature=CONFIG["TEMPERATURE"],
            max_tokens=CONFIG["MAX_TOKENS"] - 500
        )

        texto = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens

        if tokens > 8000:
            url = guardar_docx_y_retornar_link(texto, request.host_url)
            return jsonify({
                "respuesta": f"✅ El documento ha sido generado correctamente. <a href='{url}' target='_blank'>Haz clic aquí para descargarlo en Word (.docx)</a>.",
                "tokens_usados": { "total_tokens": tokens },
                "biografia_juridica": biografia_juridica
            })

        return jsonify({
            "respuesta": texto,
            "tokens_usados": { "total_tokens": tokens },
            "biografia_juridica": biografia_juridica
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ============= WEBSEARCH PRINCIPAL =============

@app.route("/envtest", methods=["GET"])
def env_test():
    return jsonify({
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "SEARCH_ENGINE_ID": os.getenv("SEARCH_ENGINE_ID")
    })


GOOGLE_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_CX")


from llama_index.schema import Document  # ✅ Importación correcta para tu versión

import os
import requests
from bs4 import BeautifulSoup
from flask import request, jsonify
from llama_index import VectorStoreIndex, Document
import traceback

SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")  # Asegúrate de definirlo en tu entorno

@app.route("/websearch", methods=["POST"])
def websearch():
    try:
        pregunta = request.get_json().get("pregunta", "").strip()
        if not pregunta:
            return jsonify({"error": "Se requiere 'pregunta'"}), 400

        # ===== Buscar en índice de respuestas prácticas =====
        index_respuestas_abogados = pc.Index("indice-respuestas-abogados")
        embedding = embed_model._get_query_embedding(pregunta)

        similares = index_respuestas_abogados.query(
            vector=embedding,
            top_k=1,
            include_metadata=True
        )

        mostrar_respuesta_practica = False
        mostrar_google = True
        respuesta_practica_html = ""

        score = 0
        if similares.get("matches"):
            match = similares["matches"][0]
            score = match.get("score", 0)

            if score >= 0.70:
                mostrar_respuesta_practica = True
                mostrar_google = False
            elif 0.60 <= score < 0.70:
                mostrar_respuesta_practica = True
                mostrar_google = True
            else:
                mostrar_google = True

            if mostrar_respuesta_practica:
                respuesta_practica = obtener_respuesta_practica(pregunta, score=score)
                if respuesta_practica:
                    respuesta_practica_html = f"<h3>📌 Recomendación práctica</h3><div class='chat-respuesta'>{respuesta_practica}</div>"
                    if mostrar_google:
                        respuesta_practica_html += "<hr>"

        respuesta_google_html = ""
        tokens = 0

        if mostrar_google:
            # ===== Buscar en Google =====
            search_url = "https://www.googleapis.com/customsearch/v1"
            search_params = {
                "key": GOOGLE_API_KEY,
                "cx": SEARCH_ENGINE_ID,
                "q": pregunta,
                "siteSearch": "gob.ec",
                "siteSearchFilter": "i"
            }
            search_resp = requests.get(search_url, params=search_params)
            search_resp.raise_for_status()
            resultados = search_resp.json().get("items", [])[:5]

            if not resultados:
                respuesta_google_html = "<h3>🌐 Respuesta basada en Google</h3><div class='chat-respuesta'>❌ No se encontraron páginas relevantes.</div>"
            else:
                contexto = "\n".join(
                    [f"Título: {item['title']}\nLink: {item['link']}" for item in resultados]
                )

                prompt = f"""
Eres un abogado experto. Un usuario te hace la siguiente pregunta:

\"{pregunta}\"

Google te ha mostrado los siguientes resultados relacionados:

{contexto}

Tu tarea es:

1. Responder al usuario de manera clara y directa, en un párrafo.
2. Indicar que debe realizar el proceso en fuentes oficiales.
3. Recomendar los enlaces más útiles (basado en los títulos) ordenados por relevancia para que el usuario continúe su trámite.

Responde con lenguaje humano, sin tecnicismos innecesarios y sin inventar datos que no estén respaldados por los títulos mostrados.
""".strip()

                response = openai_client.chat.completions.create(
                    model=CONFIG["OPENAI_MODEL"],
                    messages=[
                        {"role": "system", "content": "Eres un abogado experto que ayuda con trámites legales en Ecuador."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=CONFIG["TEMPERATURE"],
                    max_tokens=800
                )

                respuesta_ia_cruda = response.choices[0].message.content.strip()
                tokens = response.usage.total_tokens

                enlaces_html = []
                for i, item in enumerate(resultados, 1):
                    titulo = item.get("title", f"Fuente {i}").strip()
                    url = item.get("link", "#")
                    enlaces_html.append(f"<li>🔗 <strong>{titulo}</strong><br><a href='{url}' target='_blank'>{url}</a></li>")

                if enlaces_html:
                    respuesta_ia_cruda += "<br><br><ul class='fuentes-web'>" + "\n".join(enlaces_html) + "</ul>"

                respuesta_google_html = f"<h3>🌐 Respuesta basada en Google</h3><div class='chat-respuesta'>{respuesta_ia_cruda}</div>"

        # ========== RESPUESTA FINAL ==========
        return jsonify({
            "respuesta": respuesta_practica_html + respuesta_google_html,
            "tokens_usados": {"total_tokens": tokens},
            "biografia_juridica": ""
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ============= ENDPOINT PRINCIPAL =============
@app.route("/test-contexto-practico", methods=["POST"])
def test_contexto_practico():
    try:
        data = request.get_json()
        question = data.get("question", "").strip() if data else ""
        if not question:
            return jsonify({"error": "Se requiere 'question'"}), 400

        index_respuestas_abogados = pc.Index("indice-respuestas-abogados")
        embedding = embed_model._get_query_embedding(question)

        similares = index_respuestas_abogados.query(
            vector=embedding,
            top_k=1,
            include_metadata=True
        )

        if not similares.get("matches"):
            return jsonify({"respuesta": "❌ No se encontró coincidencia."})

        match = similares["matches"][0]
        score = match["score"]
        idea = match["metadata"].get("respuesta_abogado", "")
        descripcion = match["metadata"].get("descripcion", "")

        contexto_practico = f'En casos como "{descripcion.lower()}" se suele actuar de la siguiente forma: {idea[:300]}...'

        return jsonify({
            "score": score,
            "descripcion": descripcion,
            "respuesta_abogado": idea,
            "contexto_practico": contexto_practico
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ============= probar 5 =============
import time
# --- GET /responses/testMarca (más aire + cierre automático) ---
app.view_functions.pop("responses_testMarca", None)

@app.route("/responses/testMarca", methods=["GET"])
def responses_testMarca():
    import os, re, html as htmlmod
    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
    from flask import Response
    from openai import OpenAI

    # ---------- helpers ----------
    def _safe_text(resp):
        t = (getattr(resp, "output_text", "") or "").strip()
        if t: return t
        parts = []
        for it in (getattr(resp, "output", []) or []):
            if (getattr(it, "type", "") or "").lower() == "message":
                for c in (getattr(it, "content", []) or []):
                    tx = getattr(c, "text", None)
                    if tx: parts.append(tx)
        return "\n".join(parts).strip()

    def _strip_tracking_urls(text: str) -> str:
        if not text: return text
        def _clean(u):
            try:
                p = urlparse(u)
                qs = [(k,v) for (k,v) in parse_qsl(p.query, keep_blank_values=True)
                      if not (k.lower().startswith("utm_") or k.lower() in {"gclid","fbclid","ref"})]
                return urlunparse((p.scheme,p.netloc,p.path,p.params,
                                   "&".join([f"{k}={v}" for k,v in qs]) if qs else "", p.fragment))
            except Exception:
                return u
        return re.sub(r"https?://[^\s<>\)\]\"']+", lambda m: _clean(m.group(0)), text)

    def _wrap_html(body: str, title="Registro de marca en Ecuador — SENADI"):
        style = """
<style>
:root { --ink:#0b1320; --muted:#5b6472; --brand:#1b72ff; --bg:#fff; }
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Arial}
.article{max-width:900px;margin:24px auto;padding:24px}
h1,h2,h3{line-height:1.25;margin:18px 0 10px} h1{font-size:26px} h2{font-size:22px} h3{font-size:18px}
p,li{color:#1d2430} ul,ol{padding-left:22px}
a{color:var(--brand);text-decoration:none} a:hover{text-decoration:underline}
hr{border:none;border-top:1px solid #eef1f5;margin:18px 0}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:#555}
.error{background:#fff7f7;border:1px solid #ffd6d6;border-radius:10px;padding:14px}
</style>"""
        return f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{htmlmod.escape(title)}</title>{style}</head><body><article class='article'>{body}</article></body></html>"

    def _looks_good(html: str) -> bool:
        if not html or len(re.sub(r"<[^>]+>", "", html).strip()) < 250:
            return False
        if len(re.findall(r"<h2\b", html, re.I)) < 3:
            return False
        links = re.findall(r'href=[\'"](https?://[^\'"]+)', html, re.I)
        if len(set(links)) < 2:
            return False
        official = [u for u in links if re.search(r'(derechosintelectuales\.gob\.ec|\.gob\.ec)', u)]
        return len(set(official)) >= 1

    def _looks_truncated(html: str) -> bool:
        if not html:
            return True
        if len(re.findall(r"<h2\b", html, re.I)) < 3:
            return True
        needed = [r"Fuentes\s+consultadas", r"B(ús|us)quedas\s+realizadas"]
        for n in needed:
            if not re.search(n, html, re.I):
                return True
        tail_txt = re.sub(r"<[^>]+>", "", html).strip()
        if not tail_txt or not re.search(r"[\.!?]$", tail_txt):
            return True
        return False

    def _error_card(msg: str, raw: str = "") -> str:
        det = f"<details><summary class='mono'>detalle</summary><pre class='mono'>{htmlmod.escape(raw)[:5000]}</pre></details>" if raw else ""
        return f"<div class='error'><h3>⚠️ {htmlmod.escape(msg)}</h3>{det}</div>"

    # ---------- cliente ----------
    api_key = os.getenv("OPENAI_API_KEY"); project = os.getenv("OPENAI_PROJECT")
    if not api_key:
        return Response(_wrap_html(_error_card("Falta OPENAI_API_KEY.")), mimetype="text/html; charset=utf-8")
    try:
        client = OpenAI(api_key=api_key, project=project)
    except Exception as e:
        return Response(_wrap_html(_error_card("No se pudo crear el cliente OpenAI.", str(e))), mimetype="text/html; charset=utf-8")

    # ---------- prompt (solo web) ----------
    SYSTEM = (
        "Eres un asesor experto en SENADI (Ecuador). Devuelves SOLO HTML válido en español, claro y answer-first. "
        "Usa Web Search (≤7); prioriza SENADI (.gob.ec, derechosintelectuales.gob.ec / propiedadintelectual.gob.ec) y OMPI/WIPO. "
        "Reglas: "
        "• Cada monto o plazo DEBE tener un enlace oficial; si no hay fuente clara, escribe '—' y agrega 'Cómo verificar' (ruta exacta en el sitio). "
        "• Prohibidos rangos 'orientativos' para tasas oficiales. "
        "• Estructura esperada: "
        "<h2>🧭 Resumen rápido</h2>"
        "<h2>Pasos oficiales</h2>"
        "<h2>Costos y tasas</h2>"
        "<h2>Plazos</h2>"
        "<h2>📚 Fuentes consultadas</h2> (5–7 enlaces, dominios únicos, preferencia oficial)"
        "<h2>🔎 Búsquedas realizadas (≤7)</h2> (query + URL principal). "
        "• IMPORTANTE: Entrega el HTML final en ESTE MISMO TURNO; no termines en una llamada de herramienta."
    )
    USER = (
        "Pasos, costos y plazos para registrar una marca en Ecuador (SENADI). "
        "Incluye Casillero virtual, Solicitudes en línea (signos distintivos), Gaceta/oposiciones (30 días hábiles), "
        "Clasificación de Niza (OMPI), vigencia de 10 años. "
        "No inventes montos/plazos; si no constan oficialmente, usa '—' y explica cómo verificar."
    )

    # ---------- llamada 1 (más aire) ----------
    req1 = {
        "model": "gpt-5",
        "tools": [{"type": "web_search_preview"}],
        "input": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "max_output_tokens": 3200
    }
    raw1 = ""
    try:
        r1 = client.responses.create(**req1)
        raw1 = getattr(r1, "model_dump_json", lambda: "{}")()
        html = _strip_tracking_urls(_safe_text(r1))
    except Exception as e:
        return Response(_wrap_html(_error_card("Error en Responses API (intento 1).", str(e))), mimetype="text/html; charset=utf-8")

    # ---------- si no hay texto: cierre sin herramientas (más aire) ----------
    if not html:
        try:
            r2 = client.responses.create(
                model="gpt-5",
                input=[
                    {"role":"system","content":"Entrega YA la respuesta final en HTML válido (máx 1200 palabras), SIN usar herramientas, citando solo lo encontrado en las búsquedas previas."},
                    {"role":"user","content": USER},
                ],
                max_output_tokens=1800  # FIX: keyword argument correcto
            )
            html = _strip_tracking_urls(_safe_text(r2))
        except Exception as e:
            return Response(_wrap_html(_error_card("No hubo texto y falló el cierre sin herramientas.", str(e))), mimetype="text/html; charset=utf-8")

    # ---------- si luce incompleto: repair-pass (con web y más aire) ----------
    if not _looks_good(html):
        try:
            r3 = client.responses.create(
                model="gpt-5",
                tools=[{"type":"web_search_preview"}],
                input=[
                    {"role":"system","content": SYSTEM},
                    {"role":"user","content": USER},
                    {"role":"assistant","content": html[:6000]},
                    {"role":"user","content": "Mejora y completa el HTML de acuerdo a las reglas (enlaces oficiales, secciones, '—' si falta dato oficial). Devuelve SOLO HTML final."},
                ],
                max_output_tokens=2400
            )
            html2 = _strip_tracking_urls(_safe_text(r3))
            if _looks_good(html2):
                html = html2
        except Exception:
            pass

    # ---------- cierre automático si quedó truncado ----------
    if _looks_truncated(html):
        try:
            r4 = client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system",
                     "content": ("Vas a FINALIZAR el HTML ya generado. "
                                 "Devuelve SOLO el FRAGMENTO FALTANTE (no repitas lo anterior, "
                                 "no envuelvas con <html>/<body>). "
                                 "Completa '📚 Fuentes consultadas' y '🔎 Búsquedas realizadas (≤7)' si faltan, "
                                 "agrega enlaces oficiales y cierra oraciones.")},
                    {"role": "assistant", "content": html[:6000]},
                    {"role": "user", "content": "Continúa justo debajo de lo último que escribiste; solo el cierre/fragmento que falta."},
                ],
                max_output_tokens=1200
            )
            tail = _strip_tracking_urls(_safe_text(r4))
            if tail:
                tail = re.sub(r"(?is)</?(html|head|body)[^>]*>", "", tail).strip()
                if len(re.sub(r"<[^>]+>", "", tail).strip()) > 50:
                    html = (html.rstrip() + ("\n" if not html.endswith("\n") else "") + tail).strip()
        except Exception:
            pass

    # ---------- entrega ----------
    if not html:
        return Response(_wrap_html(_error_card("No se pudo generar contenido final solo con fuentes web en este momento.", raw1)), mimetype="text/html; charset=utf-8")

    return Response(_wrap_html(html), mimetype="text/html; charset=utf-8")





# Evita colisión si ya existía
app.view_functions.pop("responses_toolcheckFinal", None)

@app.route("/responses/toolcheckFinal", methods=["GET"])
def responses_toolcheckFinal():
    """
    HTML ejecutivo para cualquier pregunta gerencial (Ecuador/LATAM) con Web Search (≤7).
    - q=...        : pregunta (default: 'como registrar una marca en ecuador')
    - preview=1    : usa web_search_preview (si no, web_search)
    - model=gpt-5  : modelo (default gpt-5)
    - max=1600     : tokens de salida aprox

    Devuelve: text/html con secciones: Resumen, Pasos/Plan, Costos/Plazos (si aplica),
    Riesgos & Compliance, KPIs/Checklist (si aplica), Fuentes consultadas (5–7),
    y Búsquedas realizadas (≤7).
    """
    import os, re, json, traceback
    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
    from flask import request, Response
    from openai import OpenAI

    # ---------- Helpers ----------
    def _safe_text(resp):
        """Extrae texto aunque output_text venga vacío."""
        try:
            txt = (getattr(resp, "output_text", "") or "").strip()
            if txt:
                return txt
        except Exception:
            pass
        parts = []
        try:
            for it in (getattr(resp, "output", []) or []):
                if (getattr(it, "type", "") or "").lower() == "message":
                    for c in (getattr(it, "content", []) or []):
                        t = getattr(c, "text", None)
                        if t:
                            parts.append(t)
        except Exception:
            pass
        return "\n".join([p for p in parts if p]).strip()

    def _resp_to_dict(resp):
        for attr in ("model_dump_json", "json"):
            fn = getattr(resp, attr, None)
            if callable(fn):
                try:
                    return json.loads(fn())
                except Exception:
                    pass
        try:
            d = {}
            for k, v in resp.__dict__.items():
                try:
                    json.dumps(v); d[k] = v
                except Exception:
                    d[k] = str(v)
            return d
        except Exception:
            return {"_raw": str(resp)}

    def _extract_tool_queries(resp):
        """Reconstruye queries de los tool-calls de web_search (máx. 7)."""
        queries = []
        for it in (getattr(resp, "output", []) or []):
            t = (getattr(it, "type", "") or "").lower()
            if "web" in t and "search" in t and "call" in t:
                # Raspar posibles campos de argumentos
                candidates = []
                for key in ("arguments", "args", "input", "parameters", "tool_input", "tool_args"):
                    if hasattr(it, key):
                        candidates.append(getattr(it, key))
                if not candidates and hasattr(it, "__dict__"):
                    dct = dict(it.__dict__)
                    for key in ("arguments", "args", "input", "parameters", "tool_input", "tool_args"):
                        if key in dct:
                            candidates.append(dct[key])

                q_val = None
                for c in candidates:
                    try:
                        if isinstance(c, str):
                            obj = json.loads(c)
                        elif isinstance(c, dict):
                            obj = c
                        else:
                            obj = None
                        if isinstance(obj, dict):
                            for k in ("query", "q", "search_query"):
                                if k in obj and isinstance(obj[k], str) and obj[k].strip():
                                    q_val = obj[k].strip()
                                    break
                    except Exception:
                        if isinstance(c, str) and c.strip():
                            q_val = c.strip()
                    if q_val:
                        break
                queries.append(q_val or "—")

        # Dedupe preservando orden y corta a 7
        seen, uniq = set(), []
        for q in queries:
            key = (q or "—").strip().lower()
            if key not in seen:
                seen.add(key)
                uniq.append(q)
        return uniq[:7]

    def _strip_tracking_params_in_text(text: str) -> str:
        """Quita utm_* y similares de URLs dentro del HTML o Markdown."""
        if not text: return text
        def _clean_url(u):
            try:
                p = urlparse(u)
                qs = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True)
                      if not (k.lower().startswith("utm_") or k.lower() in {"fbclid","gclid","mc_eid","mc_cid","ref"})]
                new_q = urlencode(qs, doseq=True)
                return urlunparse((p.scheme, p.netloc, p.path, p.params, new_q, p.fragment))
            except Exception:
                return u
        url_rx = re.compile(r"https?://[^\s<>\)\]\"']+")
        return url_rx.sub(lambda m: _clean_url(m.group(0)), text)

    def _looks_html(s: str) -> bool:
        return bool(s and re.search(r"</?(h\d|p|ul|ol|li|div|section|article|table|thead|tbody|tr|td|th|strong|em|a)\b", s, re.I))

    def _markdownish_to_html(s: str) -> str:
        """Conversión ligera si el modelo devolviera Markdown."""
        if not s: return ""
        # enlaces [texto](url)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
        # **negritas** y *itálicas*
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
        # encabezados simples
        s = re.sub(r"^# (.+)$", r"<h2>\1</h2>", s, flags=re.M)
        s = re.sub(r"^## (.+)$", r"<h3>\1</h3>", s, flags=re.M)
        # bullets
        lines = s.splitlines()
        html_lines, in_ul = [], False
        for ln in lines:
            if re.match(r"^\s*[-•]\s+", ln):
                if not in_ul:
                    html_lines.append("<ul>")
                    in_ul = True
                html_lines.append("<li>" + re.sub(r"^\s*[-•]\s+", "", ln) + "</li>")
            else:
                if in_ul:
                    html_lines.append("</ul>")
                    in_ul = False
                if ln.strip():
                    html_lines.append(f"<p>{ln}</p>")
        if in_ul:
            html_lines.append("</ul>")
        return "\n".join(html_lines)

    def _ensure_html(body: str, title: str = "Asesor Ejecutivo") -> str:
        body = body.strip()
        if not _looks_html(body):
            body = _markdownish_to_html(body)
        style = """
<style>
:root { --ink:#0b1320; --muted:#5b6472; --brand:#1b72ff; --bg:#ffffff; }
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui, -apple-system, Segoe UI, Roboto, Arial;}
.article{max-width:880px;margin:24px auto;padding:24px;}
h1,h2,h3{line-height:1.25;margin:18px 0 10px;}
h1{font-size:26px} h2{font-size:22px} h3{font-size:18px}
p,li{color:#1d2430}
ul,ol{padding-left:22px}
.card{background:#fff;border:1px solid #e8edf3;border-radius:14px;padding:18px;margin:14px 0;box-shadow:0 1px 2px rgba(16,24,40,.03)}
a{color:var(--brand);text-decoration:none}
a:hover{text-decoration:underline}
hr{border:none;border-top:1px solid #eef1f5;margin:18px 0}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;background:#eef4ff;color:#1b4dff;font-weight:600;font-size:12px}
.mono{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; color:var(--muted)}
</style>"""
        html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
{style}
</head>
<body>
<article class="article">
{body}
</article>
</body></html>"""
        return html

    # ---------- Client ----------
    api_key = os.getenv("OPENAI_API_KEY"); project = os.getenv("OPENAI_PROJECT")
    if not api_key:
        return Response(_ensure_html("<h2>⚠️ Falta OPENAI_API_KEY</h2>"), mimetype="text/html")
    try:
        client = OpenAI(api_key=api_key, project=project)
    except Exception as e:
        return Response(_ensure_html(f"<h2>⚠️ No se pudo crear cliente OpenAI</h2><pre class='mono'>{e}</pre>"), mimetype="text/html")

    # ---------- Params ----------
    q = (request.args.get("q") or "como registrar una marca en ecuador").strip()
    model = (request.args.get("model") or "gpt-5").strip()
    preview = request.args.get("preview") == "1"
    try:
        max_out = int(request.args.get("max", "1600"))
    except Exception:
        max_out = 1600

    # ---------- Persona/Plantilla (alta calidad, adaptable) ----------
    SYSTEM = (
        "Eres un ASESOR EJECUTIVO jurídico/negocios para gerentes en Ecuador y LATAM. "
        "Devuelves SIEMPRE HTML válido (no Markdown, sin backticks). "
        "Usas Web Search solo si ayuda (tope ≤7), con preferencia por dominios oficiales y fuentes primarias. "
        "Reglas de contenido:\n"
        "• Respuesta 'answer-first' y accionable; español claro.\n"
        "• Si el tema es **trámite/ley**: incluye <h2>Costos y tasas</h2> y <h2>Plazos</h2> con montos/fechas SOLO si constan en fuentes oficiales; si no, '—'.\n"
        "• Si el tema es **estratégico/negocio**: incluye <h2>Decisiones y alternativas</h2> y <h2>Plan 30-60-90</h2>.\n"
        "• Siempre agrega <h2>📚 Fuentes consultadas</h2> con 5–7 enlaces únicos (preferir .gob.ec / reguladores / OMPI, etc.).\n"
        "• Cierra con <h2>🔎 Búsquedas realizadas (≤7)</h2> listando cada query y URL principal si la hay.\n"
        "• No inventes plazos/montos. Si no hay dato oficial claro, usa '—' y explica cómo verificar.\n"
        "• El texto final debe ser un solo bloque HTML (no termines en tool-call)."
    )
    USER = (
        f"Tarea: Redacta un informe ejecutivo para el/la gerente sobre: {q}\n"
        "Formato recomendado (adáptalo según el tema):\n"
        "<h2>🧭 Resumen rápido</h2>\n"
        "<h2>Pasos / Decisiones y alternativas</h2>\n"
        "<h2>Costos y tasas</h2>\n"
        "<h2>Plazos</h2>\n"
        "<h2>Riesgos & Compliance</h2>\n"
        "<h2>KPIs & Checklist</h2>\n"
        "<h2>📚 Fuentes consultadas</h2>\n"
        "<h2>🔎 Búsquedas realizadas (≤7)</h2>\n"
        "Incluye listas con <ul><li>…</li></ul> y enlaces clicables <a href='...'>…</a>."
    )
    tool = {"type": "web_search_preview"} if preview else {"type": "web_search"}

    req = {
        "model": model,
        "tools": [tool],
        "input": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "max_output_tokens": max_out  # No pasar temperature/top_p en este modelo
    }

    # ---------- Call 1: con Web Search ----------
    try:
        r = client.responses.create(**req)
    except Exception as e:
        tb = "\n".join(traceback.format_exc().splitlines()[-6:])
        return Response(_ensure_html(f"<h2>⚠️ Error en Responses API</h2><pre class='mono'>{e}\n{tb}</pre>"), mimetype="text/html")

    # Texto bruto
    text = _safe_text(r)
    text = _strip_tracking_params_in_text(text)

    # Si quedó vacío (tool-call), forzar cierre sin herramientas
    if not text:
        finalize_req = {
            "model": model,
            "input": [
                {"role": "system",
                 "content": ("Entrega AHORA la respuesta final en HTML válido, clara y 'answer-first' (máx 700 palabras), "
                             "SIN usar herramientas. Agrega '📚 Fuentes consultadas' y '🔎 Búsquedas realizadas (≤7)'.")},
                {"role": "user", "content": USER},
            ],
            "max_output_tokens": max_out
        }
        try:
            r2 = client.responses.create(**finalize_req)
            text = _safe_text(r2)
            text = _strip_tracking_params_in_text(text)
        except Exception as e:
            return Response(_ensure_html(f"<h2>⚠️ Fallback falló</h2><pre class='mono'>{e}</pre>"), mimetype="text/html")

    # ---------- Enriquecimiento: insertar búsquedas si faltan ----------
    def _has_busquedas_section(html: str) -> bool:
        return bool(re.search(r"b(us|ús)quedas\s+realizadas", html, re.I))

    if not _has_busquedas_section(text):
        # reconstruye queries y agrega sección al final
        queries = _extract_tool_queries(r)
        if queries:
            items = "\n".join([f"<li><strong>query:</strong> {q} <span class='mono'>| URL principal: —</span></li>" for q in queries])
            extra = f"\n<h2>🔎 Búsquedas realizadas (≤7)</h2>\n<ul>\n{items}\n</ul>\n"
            text = text.rstrip() + ("\n" if not text.endswith("\n") else "") + extra

    # ---------- Asegurar HTML + estilos y responder ----------
    body_html = text if _looks_html(text) else _markdownish_to_html(text)
    full_html = _ensure_html(body_html, title="Asesor Ejecutivo")

    return Response(full_html, mimetype="text/html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
