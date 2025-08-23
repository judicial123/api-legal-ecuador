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

    Comportamiento:
    - Prioriza SIEMPRE tus context_docs para razonar (LEGAL_FACTS) y cita inline [CÓDIGO Art. N] cuando uses esos hechos.
    - Complementa con Google CSE limitado a gob.ec para plazos/trámites/operativa.
    - Writer NO genera ⚖️ Fundamento legal (lo agregamos nosotros desde context_docs).
    - Quality gate: sin placeholders, tabla presente, ≥3 fuentes gob.ec, y (si hay context_docs) ≥2 citas inline válidas en “🧭 Respuesta ejecutiva”.
    """
    import os, json, re, html
    import requests

    model = CONFIG.get("OPENAI_MODEL", "gpt-5-mini")
    is_gpt5 = str(model).startswith("gpt-5")
    max_out = int(CONFIG.get("MAX_TOKENS", 2000))
    temperature = float(CONFIG.get("TEMPERATURE", 0.3))

    GOOGLE_API_KEY = CONFIG.get("GOOGLE_SEARCH_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_CX = CONFIG.get("GOOGLE_CX") or os.getenv("GOOGLE_CX")

    # ----------------- Helpers -----------------
    def _google_cse(q, n=5):
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
        out, seen = [], set()
        for r in results:
            u = r.get("link") or ""
            if u and (u not in seen):
                out.append(r); seen.add(u)
            if len(out) >= cap:
                break
        return out

    def _mk_web_text(snips):
        if not snips:
            return "No se hallaron fuentes oficiales en gob.ec."
        lines = []
        for r in snips:
            t = r.get("title", "Fuente")
            u = r.get("link", "")
            s = r.get("snippet", "")
            lines.append("Título: " + t + "\nURL: " + u + "\nExtracto: " + s)
        return "\n\n".join(lines)

    def _short_quote(txt, min_w=10, max_w=25):
        if not txt:
            return ""
        clean = " ".join(txt.split())
        words = clean.split(" ")
        if len(words) <= max_w:
            return clean
        end = max(min_w, min(max_w, 20))
        return " ".join(words[:end])

    def _mk_fundamento(docs):
        articulos, usados = [], []
        for d in docs or []:
            codigo = (d.get("codigo") or "").strip()
            art = (d.get("articulo") or "").strip()
            texto = (d.get("texto") or "").strip()
            if not (codigo and art and texto):
                continue
            ref = codigo + " Art. " + art
            cita = _short_quote(texto, 10, 25)
            articulos.append("- [" + ref + "] " + "“" + html.escape(cita) + "”.")
            usados.append(ref)
            if len(articulos) >= 6:
                break
        if not articulos:
            return ""
        cierre = "Me baso en [" + ", ".join(usados) + "]."
        return "⚖️ Fundamento legal\n" + "\n".join(articulos) + "\n\n" + cierre

    def _ensure_sections(txt, must_sections):
        missing = []
        for h in must_sections:
            if re.search(r"^" + re.escape(h) + r"\s*$", txt, flags=re.IGNORECASE | re.MULTILINE) is None:
                missing.append(h)
        if "| Obligación | Plazo legal | Responsable | Fuente" not in txt:
            missing.append("🗓️ Tabla de plazos y responsables (tabla Markdown)")
        return missing

    def _has_forbidden(txt):
        forbidden = [
            "Lorem ipsum", "No se puede proporcionar", "información adicional",
            "Departamento X", "Departamento Y", "Fuente 1", "Fuente 2"
        ]
        return any(f.lower() in txt.lower() for f in forbidden)

    # ----------------- Preparar LEGAL_FACTS (prioridad #1) -----------------
    legal_facts_lines, allowed_refs = [], set()
    for d in (context_docs or [])[:12]:
        cod = (d.get("codigo") or "").strip()
        art = (d.get("articulo") or "").strip()
        txt = (d.get("texto") or "").strip()
        if cod and art and txt:
            ref = cod + " Art. " + art
            allowed_refs.add(ref)
            legal_facts_lines.append(ref + ": " + txt[:600])
    legal_facts_text = "\n".join(legal_facts_lines) if legal_facts_lines else "(sin documentos)"

    # ----------------- (1) PROMPT BUILDER -----------------
    builder_system = (
        "Eres un “Prompt Architect” legal para Ecuador. Convierte una pregunta de negocio en un encargo perfecto "
        "para un abogado corporativo (gerencia/RRHH/contabilidad).\n"
        "Reglas:\n"
        "- Incluir búsqueda web en dominios oficiales gob.ec.\n"
        "- Maximiza utilidad: answer-first explícito, pasos, plazos, multas, checklist, errores comunes.\n"
        "- Si el texto sugiere urgencia temporal, timeline_required=true.\n"
        "- Si un dato no consta en fuentes, usa “—”.\n"
        "Devuelve SOLO JSON válido con: queries_gob_ec (3–6), must_rules (4–8), answer_hints (3–6), timeline_required (bool)."
    )
    builder_user = 'user_question: """' + (question or "") + '"""'

    kwargs_b = dict(model=model, messages=[
        {"role": "system", "content": builder_system},
        {"role": "user", "content": builder_user}
    ])
    if is_gpt5:
        kwargs_b["max_completion_tokens"] = 350
    else:
        kwargs_b["temperature"] = max(0.2, temperature - 0.1)
        kwargs_b["max_tokens"] = 350

    tokens_total = 0
    try:
        br = openai_client.chat.completions.create(**kwargs_b)
        builder_txt = (br.choices[0].message.content or "").strip()
        tokens_total += br.usage.total_tokens if getattr(br, "usage", None) else 0
        try:
            bj = json.loads(builder_txt)
        except Exception:
            bj = {}
    except Exception:
        bj = {}

    base_q = (question or "")
    queries = bj.get("queries_gob_ec") or [
        base_q,
        base_q + " sitio:gob.ec",
        base_q + " Ministerio del Trabajo SUT site:trabajo.gob.ec",
        base_q + " IESS site:iess.gob.ec",
        base_q + " SRI formulario site:sri.gob.ec",
    ]
    must_rules = bj.get("must_rules") or [
        "Responder explícitamente la pregunta en la primera sección.",
        "No inventar plazos ni montos; usar “—” si no consta.",
        "Citar sólo dominios gob.ec en 📚 Fuentes consultadas.",
        "Entregar tabla de plazos y responsables.",
        "Incluir checklist y errores comunes.",
        "Cuando uses LEGAL_FACTS, añade referencia [CÓDIGO Art. N] al final de la frase."
    ]
    answer_hints = bj.get("answer_hints") or []
    timeline_required = bool(bj.get("timeline_required", False))

    # ----------------- (2) WEB CSE (prioridad #2) -----------------
    all_results = []
    for q in queries[:6]:
        if q and isinstance(q, str):
            all_results.extend(_google_cse(q, n=4))
    web_snippets = _mk_snippets(all_results, cap=8)
    web_context_text = _mk_web_text(web_snippets)

    # ----------------- (3) ANSWER WRITER -----------------
    required_sections = [
        "🧭 Respuesta ejecutiva",
        "🗓️ Tabla de plazos y responsables",
        "💸 Multas y consecuencias",
        "✅ Acciones inmediatas",
        "🧾 Checklist de documentos",
        "❌ Errores comunes",
        "📚 Fuentes consultadas"
    ]
    writer_system = (
        "Eres un abogado corporativo ecuatoriano para gerentes.\n"
        "Tarea: responder explícitamente la pregunta y entregar una GUÍA EJECUTIVA completa.\n"
        "Prioridad de fuentes:\n"
        "1) LEGAL_FACTS (tus artículos). Cuando uses un hecho de LEGAL_FACTS, cierra la frase con [CÓDIGO Art. N].\n"
        "2) WEB_SNIPPETS (gob.ec) para plazos, trámites, operativa y ejemplos.\n"
        "Reglas duras:\n"
        "- Si un dato no aparece en LEGAL_FACTS ni en WEB_SNIPPETS, escribe “—”.\n"
        "- Prohibido ‘rellenos’: “No se puede proporcionar…”, “Lorem ipsum…”, “Departamento X/Y”, “Fuente 1/2”.\n"
        "- Cita fuentes al final (📚) con Título + dominio (sin repetir dominios). No generes ‘⚖️ Fundamento legal’.\n"
        "- Tono profesional y directo. Si TIMELINE_REQUIRED=true, organiza acciones por Día 1… Día 7 en “✅ Acciones inmediatas”.\n"
        "Salida: usa EXACTAMENTE estos encabezados (en este orden):\n"
        + "\n".join(required_sections)
    )

    sections_text = "\n".join(["- " + s for s in required_sections])
    must_rules_text = "\n".join(["- " + r for r in must_rules])
    hints_text = ("\n".join(["- " + h for h in answer_hints])) if answer_hints else "- —"

    writer_user = (
        "PREGUNTA:\n" + (question or "") +
        "\n\nLEGAL_FACTS (prioridad #1, cita como [CÓDIGO Art. N] cuando apliques):\n" + legal_facts_text +
        "\n\nWEB_SNIPPETS (prioridad #2, sólo gob.ec):\n" + web_context_text +
        "\n\nSECTIONS (usar exactamente estos encabezados):\n" + sections_text +
        "\n\nMUST_RULES:\n" + must_rules_text +
        "\n\nANSWER_HINTS:\n" + hints_text +
        "\n\nTIMELINE_REQUIRED: " + ("true" if timeline_required else "false") +
        "\n\nNOTA: El bloque ‘⚖️ Fundamento legal’ lo añadirá el sistema después, con citas textuales de LEGAL_FACTS."
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
    tokens_total = (wr.usage.total_tokens if getattr(wr, "usage", None) else 0) + tokens_total

    # ----------------- (4) Quality Gate -----------------
    missing = _ensure_sections(respuesta, required_sections)
    too_few_sources = ("📚 Fuentes consultadas" in respuesta) and (len(re.findall(r"^- ", respuesta.split("📚 Fuentes consultadas",1)[-1], flags=re.MULTILINE)) < 3)
    bad_text = _has_forbidden(respuesta)

    # Si hay context_docs, requerimos ≥2 citas inline válidas en “🧭 Respuesta ejecutiva”
    need_inline_legal = False
    if context_docs and allowed_refs:
        exec_block = respuesta.split("🧭 Respuesta ejecutiva", 1)
        inline_ok = 0
        if len(exec_block) > 1:
            tail = exec_block[1]
            # Busca referencias [CÓDIGO Art. N] y verifica que existan en allowed_refs
            for m in re.findall(r"\[([^\[\]]+? Art\. ?\d+)\]", tail):
                if m.strip() in allowed_refs:
                    inline_ok += 1
        if inline_ok < 2:
            need_inline_legal = True

    if missing or too_few_sources or bad_text or need_inline_legal:
        problems = []
        if missing: problems.append("Faltan secciones: " + ", ".join(missing))
        if too_few_sources: problems.append("Hay menos de 3 fuentes en 📚.")
        if bad_text: problems.append("Se detectaron frases prohibidas/‘relleno’.")
        if need_inline_legal: problems.append("Faltan ≥2 citas inline [CÓDIGO Art. N] válidas en ‘🧭 Respuesta ejecutiva’.")
        fix_instr = (
            "Corrige para cumplir EXACTAMENTE los encabezados requeridos y las reglas duras.\n"
            "- Usa LEGAL_FACTS como fuente principal; cuando apliques un hecho legal, cierra la frase con [CÓDIGO Art. N].\n"
            "- Asegura ≥2 citas inline válidas en ‘🧭 Respuesta ejecutiva’.\n"
            "- Prohibido texto de relleno (‘No se puede proporcionar…’, ‘Lorem ipsum…’, ‘Departamento X/Y’, ‘Fuente 1/2’).\n"
            "- Asegura la tabla y al menos 3 fuentes (dominios gob.ec). Emplea “—” si falta un dato.\n"
            "- NO generes ‘⚖️ Fundamento legal’. Ese bloque lo añadirá el sistema."
        )
        fix_user = (
            "; ".join(problems) + "\n\n"
            "PREGUNTA:\n" + (question or "") +
            "\n\nLEGAL_FACTS (prioridad #1):\n" + legal_facts_text +
            "\n\nWEB_SNIPPETS (gob.ec):\n" + web_context_text +
            "\n\nSECTIONS requeridos (no cambies títulos):\n" + "\n".join(["- " + s for s in required_sections])
        )
        kwargs_fix = dict(model=model, messages=[
            {"role": "system", "content": fix_instr},
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

    # ----------------- (5) Notas operativas opcionales -----------------
    if contexto_practico:
        respuesta += "\n\n🧩 Notas operativas (no normativo)\n" + (contexto_practico.strip()[:1200])

    # ----------------- (6) Fundamento legal desde tus docs -----------------
    if context_docs:
        bloque = _mk_fundamento(context_docs)
        if bloque:
            # Si el Writer metió uno (no debería), lo reemplazamos
            if "⚖️ Fundamento legal" in respuesta:
                respuesta = re.split(r"\n?⚖️ Fundamento legal.*", respuesta, maxsplit=1)[0].rstrip()
            respuesta += "\n\n" + bloque

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
        respuesta_legal, tokens_usados = generate_legal_response_empresario(question, context_docs)

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



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
