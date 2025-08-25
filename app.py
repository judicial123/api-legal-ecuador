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
    Versión playground-like:
    - No usa context_docs ni añade ⚖️ Fundamento legal.
    - Delegado a Responses API + Web Search (GPT-5 mini).
    - Estructura práctica y enlaces como en Playground.
    """
    import re, html, logging
    from urllib.parse import urlparse

    # -------- Config (FORZAR GPT-5 mini) --------
    CONFIG = globals().get("CONFIG", {}) if "CONFIG" in globals() else {}
    model = CONFIG.get("OPENAI_MODEL_RESPONSES") or CONFIG.get("OPENAI_MODEL") or "gpt-5-mini"
    if not str(model).startswith("gpt-5"):
        model = "gpt-5-mini"
    max_out = int(CONFIG.get("MAX_TOKENS", 3000))
    temperature = 0.3
    logging.getLogger().warning(f"[Empresario_API|Playground] Using Responses model: {model}")

    # -------- Cliente (Responses API) --------
    try:
        from openai import OpenAI
        client = globals().get("openai_client", None)
        if client is None or not hasattr(client, "responses"):
            client = OpenAI()
    except Exception as e:
        return ("⚠️ Falta SDK OpenAI (instala `openai>=1.40`). Detalle: " + str(e), 0)

    # -------- Helpers --------
    def _mk_fuentes_html_from_urls(urls, titles=None, cap=7):
        titles = titles or {}
        used_domains, items = set(), []
        for u in urls:
            if not u:
                continue
            try:
                d = urlparse(u).netloc.replace("www.","")
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
        """Intenta recuperar URLs/títulos desde los eventos del Responses API; si no, regex del texto."""
        urls, titles = [], {}
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
        text = (getattr(resp, "output_text", "") or "")
        if not urls and text:
            for m in re.findall(r"https?://[^\s\)\]\}<>\"']+", text):
                if m not in urls:
                    urls.append(m)
        return urls, titles

    # -------- Prompt estilo Playground (sin LEGAL_FACTS) --------
    q = (question or "").strip()

    SYSTEM = (
        "Eres un asesor experto en trámites empresariales en Ecuador (es-EC, tz: America/Guayaquil). "
        "Usa la herramienta Web Search de forma ACTIVA (varias búsquedas y contraste) priorizando dominios oficiales "
        "(supercias.gob.ec, sri.gob.ec, gob.ec, funcionjudicial.gob.ec, registros mercantiles). "
        "Responde answer-first y NO inventes montos/plazos: si no están claros, escribe “—” y explica cómo obtenerlos. "
        "FORMATO EXACTO (máx 8 secciones):\n"
        "1) 🧭 Respuesta ejecutiva (bullets cortos con veredicto claro: Sí/No/Depende…)\n"
        "2) Pasos oficiales\n"
        "3) Documentos mínimos\n"
        "4) Costos y tasas (si hay)\n"
        "5) Plazos típicos (si hay)\n"
        "6) ✅ Acciones inmediatas / ❌ Errores comunes (opcional)\n"
        "7) 🧩 Tips operativos (opcional)\n"
        "8) 📚 Fuentes consultadas (obligatorio) como <ul> 5–7 enlaces (título + dominio), sin repetir dominio.\n"
        "Nunca dejes la respuesta sin texto final; si aún estás buscando, escribe una síntesis útil igualmente."
    )

    USER = (
        "OBJETIVO: igualar el estilo del Playground con pasos oficiales, documentos mínimos, costos/tasas y plazos, con enlaces oficiales.\n\n"
        "PREGUNTA:\n" + q
    )

    tools = [{"type": "web_search"}]
    tokens_total = 0

    # -------- Llamada principal (con web_search) --------
    try:
        r = client.responses.create(
            model=model,
            input=[{"role":"system","content":SYSTEM},
                   {"role":"user","content":USER}],
            tools=tools,
            max_output_tokens=max_out,
            temperature=temperature,
        )
    except Exception as e:
        # Fallback a web_search_preview si existe en el tenant
        try:
            r = client.responses.create(
                model=model,
                input=[{"role":"system","content":SYSTEM},
                       {"role":"user","content":USER}],
                tools=[{"type":"web_search_preview"}],
                max_output_tokens=max_out,
                temperature=temperature,
            )
        except Exception as e2:
            aviso = "ℹ️ Nota: tu cuenta no tiene habilitado Web Search en Responses API; el resultado puede diferir del Playground."
            return (aviso, 0)

    # -------- Salida del modelo --------
    respuesta = getattr(r, "output_text", "") or ""
    try:
        u = getattr(r, "usage", None)
        tokens_total = (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0)
    except Exception:
        tokens_total = 0

    # -------- Si no devolvió texto, reintento sin herramientas (síntesis mínima) --------
    if not respuesta.strip():
        r2 = client.responses.create(
            model=model,
            input=[
                {"role":"system","content":"Responde en el formato pedido, AUN SIN BUSCAR, usando “—” donde falten datos, y SIEMPRE incluye '📚 Fuentes consultadas' con una lista vacía si no tienes enlaces."},
                {"role":"user","content":USER}
            ],
            max_output_tokens=min(max_out, 800),
            temperature=0.2,
        )
        respuesta = getattr(r2, "output_text", "") or ""
        try:
            u2 = getattr(r2, "usage", None)
            tokens_total += (getattr(u2, "input_tokens", 0) or 0) + (getattr(u2, "output_tokens", 0) or 0)
        except Exception:
            pass

    # -------- Forzar sección de Fuentes si falta --------
    if "📚 Fuentes consultadas" not in respuesta:
        urls, titles = _extract_citations(r)
        fuentes_html = _mk_fuentes_html_from_urls(urls, titles, cap=7)
        respuesta = respuesta.rstrip() + "\n\n📚 Fuentes consultadas\n" + fuentes_html
    else:
        # Si existe la sección pero sin <ul>, la completamos con las citas detectadas
        if "<ul>" not in respuesta:
            urls, titles = _extract_citations(r)
            fuentes_html = _mk_fuentes_html_from_urls(urls, titles, cap=7)
            # Reemplaza el bloque de fuentes por nuestra lista HTML
            respuesta = re.sub(r"(📚\s*Fuentes consultadas\s*)[\s\S]*$", r"\1\n" + fuentes_html, respuesta)

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


import time

@app.route("/gpt5/test2", methods=["GET"])
def gpt5_test():
    try:
        prompt = request.args.get("q", "Di 'pong' y el nombre exacto del modelo que estás usando.")
        t0 = time.time()
        resp = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Eres un probador mínimo. Responde en una sola línea."},
                {"role": "user", "content": prompt}
            ],
            # GPT-5: NO usar temperature; usa el default del modelo
            max_completion_tokens=50  # reemplaza max_tokens por max_completion_tokens
        )
        latency_ms = int((time.time() - t0) * 1000)
        content = resp.choices[0].message.content.strip()
        model_used = getattr(resp, "model", "gpt-5-mini")
        tokens = resp.usage.total_tokens if getattr(resp, "usage", None) else None

        return jsonify({"ok": True, "model": model_used, "latency_ms": latency_ms,
                        "total_tokens": tokens, "sample": content})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/responses/toolcheck", methods=["GET"])
def responses_toolcheck():
    """
    Verifica si Web Search está habilitado para Responses API con gpt-5-mini.
    - Paso 1 (ping): intenta una llamada con tools=[{"type": "web_search"}] para confirmar que el endpoint acepta la herramienta.
    - Paso 2 (uso): instruye a hacer UNA búsqueda y luego responder 'ok'; intenta detectar si hubo señales de uso de la herramienta.
    """
    from openai import OpenAI
    from flask import jsonify

    try:
        # Cliente OpenAI (usa el global si existe)
        client = globals().get("openai_client") or OpenAI(api_key=CONFIG.get("OPENAI_API_KEY"))

        model = "gpt-5-mini"  # Fijamos el modelo a probar

        # ----- Paso 1: ping (¿el endpoint acepta la herramienta?) -----
        try:
            ping = client.responses.create(
                model=model,
                input=[{"role": "user", "content": "di 'ok'"}],
                tools=[{"type": "web_search"}],
                max_output_tokens=32,  # >= 16 para evitar error de mínimo
            )
            ping_model = getattr(ping, "model", model)
        except Exception as e_ping:
            return jsonify({
                "ok": False,
                "model_requested": model,
                "web_search_enabled": False,  # No aceptó la herramienta
                "error": str(e_ping),
            }), 200

        # ----- Paso 2: intento de uso (sin tool_choice; solo instrucción) -----
        try:
            r = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "Haz exactamente UNA búsqueda web y luego responde solo con 'ok'."},
                    {"role": "user", "content": "Encuentra la página oficial del Servicio de Rentas Internas de Ecuador (SRI) usando la herramienta y luego responde 'ok'."}
                ],
                tools=[{"type": "web_search"}],
                max_output_tokens=64
            )

            # Extraer señales de uso (heurística sobre metadatos en la salida estructurada)
            used = False
            try:
                for item in (getattr(r, "output", []) or []):
                    meta = None
                    if isinstance(item, dict):
                        meta = item.get("citation") or item.get("metadata") or item.get("source")
                    else:
                        meta = getattr(item, "citation", None) or getattr(item, "metadata", None)
                    if meta:
                        u = (meta.get("url") or meta.get("href") or meta.get("source_url") or "")
                        if u:
                            used = True
                            break
            except Exception:
                used = False

            usage = getattr(r, "usage", None)
            return jsonify({
                "ok": True,
                "model_used": getattr(r, "model", ping_model),
                "web_search_enabled": True,               # El ping pasó: la herramienta está habilitada
                "web_search_used_in_call": used,          # Señal (heurística) de uso real en esta llamada
                "output": getattr(r, "output_text", ""),
                "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            }), 200

        except Exception as e_use:
            # La herramienta está habilitada (pasó el ping), pero esta llamada falló por otro motivo
            return jsonify({
                "ok": True,
                "model_used": ping_model,
                "web_search_enabled": True,
                "web_search_used_in_call": False,
                "error_in_use_call": str(e_use),
            }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
