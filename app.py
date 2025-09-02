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
    "GOOGLE_CX": os.getenv("GOOGLE_CX"),
    "GOOGLE_CX_OFICIAL"= os.getenv("GOOGLE_CX_OFICIAL"),
    "GOOGLE_CX_PRACTICO" = os.getenv("GOOGLE_CX_PRACTICO"),
    "OPENAI_GPT5_MODEL": "gpt-5-mini",
    "MAX_COMPLETION_TOKENS_CLASSIFIER": 160,
    "MAX_OUTPUT_TOKENS_WRITER": 2000,
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


# ================== HELPERS WEB ==================
# ================== EMPRESARIO: HELPERS + ENDPOINT ==================
# (PÉGALO TAL CUAL EN TU ARCHIVO PRINCIPAL)

import time, json
import numpy as np
import re
from bs4 import BeautifulSoup
import requests

# ---- Helpers de scraping (no estaban en tu main) ----
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "es-EC,es;q=0.9"}
MENU_BLACKLIST = re.compile(
    r"(Inicio|Biblioteca|Transparencia|Noticias|Servicios Electr[oó]nicos|"
    r"Ciudadanos|Administraci[oó]n|Gobierno|Contacto|Men[uú]|Quipux)",
    re.I
)

def google_searchEmpresa(query: str, num=10, hl="es", gl="EC", start=1, cx=None):
    """
    Google Custom Search (Programmable Search Engine).
    Usa tus claves quemadas en CONFIG para evitar conflictos con env vars.
    """
    key = CONFIG.get("GOOGLE_SEARCH_API_KEY") or CONFIG.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Falta GOOGLE_SEARCH_API_KEY/GOOGLE_API_KEY en CONFIG")

    cx = cx or CONFIG.get("GOOGLE_CX")
    if not cx:
        raise RuntimeError("Falta GOOGLE_CX en CONFIG")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": key,
        "cx": cx,
        "num": num,
        "hl": hl,
        "gl": gl,
        "cr": "countryEC",
        "start": start
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return [it["link"] for it in data.get("items", [])]

def smart_scrape(url: str, max_chars=20000) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        blocks = []
        for tag in soup.find_all(["article","section","div","p","li","td"]):
            t = tag.get_text(" ", strip=True)
            if not t or len(t) < 30:
                continue
            if MENU_BLACKLIST.search(t):
                continue
            link_count = len(tag.find_all("a"))
            score = len(t) / (1 + link_count)
            blocks.append((score, t))
        blocks.sort(key=lambda x: x[0], reverse=True)
        joined = " ".join([t for _, t in blocks[:120]])
        return joined[:max_chars]
    except Exception as e:
        return f"[Error leyendo {url}: {e}]"

def chunk_text(text, max_chars=3000, overlap=250):
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur_len + len(p) + 2 <= max_chars:
            cur.append(p); cur_len += len(p) + 2
        else:
            if cur: chunks.append("\n\n".join(cur))
            tail = ("\n\n".join(cur))[-overlap:] if cur else ""
            cur = [tail, p] if tail else [p]
            cur_len = len(tail) + len(p)
    if cur: chunks.append("\n\n".join(cur))
    return [c[:max_chars] for c in chunks]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-10))

# ---- Writer final (usa Responses API, modelo GPT-5 que agregarás a CONFIG) ----
def ask_final_gptEmpresa(question, full_context, urls):
    SYSTEM = (
        "Eres un abogado empresarial en Ecuador, experto en derecho laboral, tributario y societario. "
        "Tu tono debe ser firme, claro y seguro, como si asesoraras directamente a un gerente general. "
        "Responde PREFERENTEMENTE con base en el contexto proporcionado.\n\n"
        "Instrucciones clave:\n"
        "- SI hay DOCUMENTOS LEGALES en el contexto, cítalos siempre de forma explícita con el formato [Código Art. X].\n"
        "- Si no hay normativa aplicable, responde igual con orientación práctica y no inventes artículos.\n"
        "- No ignores los artículos relevantes: son la base de la respuesta, pero no los uses si no aplican.\n"
        "- Entrega la respuesta en tono profesional y categórico, evitando frases como 'no hay suficiente información'. "
        "En su lugar, usa expresiones como 'Con base en la normativa disponible'.\n"
        "- Estructura la respuesta en bloques claros:\n"
        "   1) 🧭 Resumen ejecutivo: respuesta directa en 2–3 frases contundentes.\n"
        "   2) ✅ Recomendaciones prácticas: lista de acciones claras. (Inclúyelo solo si aplica)\n"
        "   3) ⚖️ Fundamento legal: redacta un único párrafo-resumen citando los artículos relevantes. "
        "No enumeres ni resumas artículo por artículo, solo integra lo esencial en un texto fluido. (Inclúyelo solo si aplica)\n"
        "- Cierra SIEMPRE con: 'Me baso en [artículos citados]' cuando hayas citado normativa; si no hay artículos aplicables, omite esa frase.\n"
    )
    USER = f"Pregunta: {question}\n\n{full_context}"

    try:
        r = openai_client.responses.create(
            model=CONFIG["OPENAI_GPT5_MODEL"],
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER}
            ],
            max_output_tokens=CONFIG["MAX_OUTPUT_TOKENS_WRITER"]
        )
        text = (getattr(r, "output_text", "") or "").strip()
    except Exception as e:
        text = f"❌ Error al generar respuesta: {e}"

    if not text.strip():
        text = "🧭 Resumen ejecutivo: —\n\n📚 Fuentes consultadas: —"

    if urls:
        links_html = "<h2>📚 Fuentes consultadas</h2>\n<ul>\n" + \
                     "".join(f'  <li><a href="{u}" target="_blank">{u}</a></li>\n' for u in urls) + \
                     "</ul>"
        text += "\n\n" + links_html
    return text


# ---- ENDPOINT /queryEmpresario (reutiliza tu 'index' y 'embed_model' ya creados) ----
@app.route("/queryEmpresario", methods=["POST"])
def query_empresario():
    try:
        data = request.get_json() or {}
        question_raw = (data.get("question") or "").strip()
        if not question_raw:
            return jsonify({"error": "Se requiere 'question'"}), 400

        question = re.sub(r"\s+", " ", question_raw).replace("“", '"').replace("”", '"').strip()

        # 1) Optimización + Clasificación (usa GPT-5 sin temperature)
        opt_prompt = (
            "Eres un asistente legal ecuatoriano.\n"
            "Tu tarea es reformular la pregunta de un gerente en una QUERY corta para Google Custom Search API (CSE).\n\n"
            "Reglas para la query:\n"
            "- Usa estilo palabras clave (ejemplo: 'multas empresa SRI IVA Ecuador').\n"
            "- NO uses frases completas ni preguntas.\n"
            "- Entre 3 y 6 palabras clave.\n"
            "- Siempre incluye 'Ecuador'.\n"
            "- Si es tributaria, incluye también 'SRI' y 'multa' o 'sanción'.\n"
            "- Si es laboral, incluye 'Ministerio Trabajo'.\n"
            "- Si es de marcas o propiedad intelectual, incluye 'SENADI'.\n"
            "- Si es de lavado de activos o reportes financieros, incluye 'UAFE'.\n"
            "- Evita repetir palabras innecesarias.\n"
            "- No incluyas símbolos, signos de interrogación ni conectores.\n\n"
            "Ejemplos:\n"
            "Pregunta: 'sanciones si mi empresa no presenta la declaración de IVA dentro del plazo establecido?'\n"
            "→ {\"query\":\"multas empresa SRI IVA Ecuador\", \"classification\":\"web\"}\n\n"
            "Pregunta: '¿Qué pasa si no pago el décimo tercer sueldo a mis empleados?'\n"
            "→ {\"query\":\"multa décimo tercer sueldo Ministerio Trabajo Ecuador\", \"classification\":\"web\"}\n\n"
            "Pregunta: '¿Cómo registrar la marca de mi empresa en Ecuador?'\n"
            "→ {\"query\":\"registro marca empresa SENADI Ecuador\", \"classification\":\"web\"}\n\n"
            "Pregunta: '¿Qué sanciones aplican si no reporto operaciones a la UAFE?'\n"
            "→ {\"query\":\"multas sanciones reporte UAFE Ecuador\", \"classification\":\"web\"}\n\n"
            "Devuelve SOLO JSON válido, en este formato exacto:\n"
            "{\"query\":\"...\", \"classification\":\"...\"}"
        )

        r = openai_client.chat.completions.create(
            model=CONFIG["OPENAI_GPT5_MODEL"],
            messages=[
                {"role": "system", "content": opt_prompt},
                {"role": "user", "content": question}
            ],
            max_completion_tokens=CONFIG["MAX_COMPLETION_TOKENS_CLASSIFIER"]
        )
        try:
            parsed = json.loads(r.choices[0].message.content)
            query_opt = parsed.get("query", question).strip()
            classification = parsed.get("classification", "mixto")
        except Exception:
            query_opt, classification = question, "mixto"

        if "ecuador" not in query_opt.lower():
            query_opt += " Ecuador"

        # 2) Recuperación de CÓDIGOS desde tu índice existente
        context_docs = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}

        if classification in ("codigos", "mixto"):
            retriever = index.as_retriever(similarity_top_k=12)
            results = retriever.retrieve(question)
            total_docs = len(results)
            alta_lim = int(total_docs * 0.3)
            media_lim = int(total_docs * 0.6)

            for i, rnode in enumerate(results):
                meta = getattr(rnode.node, "metadata", {})
                codigo = meta.get("code", "") or meta.get("codigo", "")
                articulo = meta.get("article", "") or meta.get("articulo", "")
                texto = getattr(rnode.node, "text", "") or meta.get("text", "")
                texto = (texto or "").strip()
                if not (codigo and articulo and texto):
                    continue
                doc = {"codigo": codigo, "articulo": articulo, "texto": texto[:600]}
                context_docs.append(doc)

                if i < alta_lim:
                    biografia_juridica["alta"].append(doc)
                elif i < media_lim:
                    biografia_juridica["media"].append(doc)
                else:
                    biografia_juridica["baja"].append(doc)

            context_docs = context_docs[:8]

        # 3) WEB (CSE oficial + práctico)
        TOPK, CHUNK_MAX, CHUNK_OVERLAP = 10, 3000, 250
        CONTEXT_HARD_CAP, MIN_CHUNKS_VALIDOS, MAX_SELECTED = 60000, 2, 12

        def rank_chunks(chunks_by_url, q):
            q_emb = embed_model.get_text_embedding(q[:2000])
            scored = []
            for u, chs in chunks_by_url:
                for idx, ch in enumerate(chs):
                    ch = (ch or "").strip()
                    if not ch:
                        continue
                    if len(ch) > CHUNK_MAX:
                        ch = ch[:CHUNK_MAX]
                    emb = embed_model.get_text_embedding(ch)
                    sim = cosine_similarity(q_emb, emb)
                    scored.append((u, idx, ch, sim))
            scored.sort(key=lambda x: x[3], reverse=True)
            return scored

        def pack_selected(selected, cap=CONTEXT_HARD_CAP):
            out, total = [], 0
            n = min(len(selected), MAX_SELECTED)
            for k in range(n):
                u, i, ch, _ = selected[k]
                piece = f"\n\n[Fuente {u} — chunk {k+1}/{n}]\n{ch}"
                if total + len(piece) > cap:
                    break
                out.append(piece)
                total += len(piece)
            return "".join(out), total

        urls, selected_urls, final_web_context = [], [], []
        if classification in ("web", "mixto"):
            urls_oficial = google_searchEmpresa(query_opt, num=TOPK, hl="es", gl="EC", cx=CONFIG["GOOGLE_CX_OFICIAL"])
            urls_practico = google_searchEmpresa(query_opt, num=TOPK, hl="es", gl="EC", cx=CONFIG["GOOGLE_CX_PRACTICO"])
            urls = urls_practico[:5] + urls_oficial[:5]

            chunks_by_url = [
                (u, chunk_text(smart_scrape(u, max_chars=20000), CHUNK_MAX, CHUNK_OVERLAP))
                for u in urls
            ]
            chunks_by_url = [(u, chs) for (u, chs) in chunks_by_url if chs]

            if chunks_by_url:
                scored = rank_chunks(chunks_by_url, question)
                if len(scored) >= MIN_CHUNKS_VALIDOS:
                    selected = scored[:MAX_SELECTED]
                    web_context, _ = pack_selected(selected)
                    final_web_context = web_context
                    selected_urls = [u for u, _, _, _ in selected]

        # 4) CONTEXTO combinado
        parts = []
        if context_docs:
            parts.append("DOCUMENTOS LEGALES:\n" + "\n".join(
                f"{d['codigo']} Art.{d['articulo']}: {d['texto']}" for d in context_docs
            ))
        if isinstance(final_web_context, str) and final_web_context.strip():
            parts.append("WEB:\n" + final_web_context)
        context_text = "\n\n".join(parts) if parts else "—"

        # 5) Respuesta final
        answer_html = ask_final_gptEmpresa(question, context_text, selected_urls)

        return jsonify({
            "respuesta": answer_html,
            "biografia_juridica": biografia_juridica,
            "tokens_usados": {"total_tokens": 0}
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
# ================== FIN EMPRESARIO ==================





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
