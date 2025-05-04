import os
import re
import traceback
import unicodedata
from flask import Flask, request, jsonify
from openai import OpenAI
from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext
from pinecone import Pinecone
from dotenv import load_dotenv
from llama_index.text_splitter import TokenTextSplitter
from llama_index import SimpleDirectoryReader
import uuid

document_store = {}


# ======================= CONFIGURACIÓN =======================
load_dotenv()

CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "INDEX_NAME": os.getenv("INDEX_NAME"),
    "PINECONE_ENV": os.getenv("PINECONE_ENV"),
    "OPENAI_MODEL": "gpt-3.5-turbo",
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 1500
}

TOP_K_RESULTS = 20
MAX_DOCS_TO_OPENAI = 6
MAX_TEXT_CHARS = 600  # <--- Aquí limitamos a OpenAI

MATH_KEYWORDS = ['calcular', 'cálculo', 'fórmula', 'sumar', 'restar', 'multiplicar', 'dividir']

# ======================= UTILS =======================
def normalizar(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    ).lower()

# ======================= APP =======================
app = Flask(__name__)

pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"], environment=CONFIG["PINECONE_ENV"])
pinecone_index = pc.Index(CONFIG["INDEX_NAME"])
openai_client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="text")
embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=CONFIG["OPENAI_API_KEY"])
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

# ======================= OPENAI CALL =======================
def generate_legal_response(question, context_docs):
    system_prompt = """
    Eres un abogado especialista en derecho ecuatoriano. Tu tarea es responder EXCLUSIVAMENTE con base en los textos legales entregados a continuación. Está TERMINANTEMENTE PROHIBIDO utilizar conocimiento externo, suposiciones, interpretaciones o completar información más allá de lo provisto.

    ⚖️ Modo de redacción:
    - Redacta con tono empático, profesional y claro.
    - Comienza con una introducción cordial, como: “Con gusto le indico que…” o “Respecto a su consulta…”.
    - Sé directo y fundamentado, evitando frases como “como experto legal…”.

    📚 Estructura requerida:
    1. Redacta una respuesta explicativa clara, pero cada afirmación o dato debe mencionar expresamente de qué artículo y código o ley proviene.
    2. Incluye al final una lista de los artículos citados: número y nombre del código o ley.
    3. Incluye citas textuales legales, incluso si están truncadas por el contexto.
    4. Finaliza siempre con la frase: “Me baso en [artículos citados]”.

    ⚠️ Reglas estrictas:
    - NO cites artículos, códigos o leyes que no estén literal y explícitamente incluidos en el contexto proporcionado.
    - NO utilices jurisprudencia, doctrina, interpretación propia ni sentido común.
    - NO completes ideas ni “corrijas” el texto legal aunque te parezca incompleto.
    - Si no hay normativa aplicable: responde exactamente “No encontré normativa aplicable. No me baso en ningún artículo.”
    """



    context_text = "\nDOCUMENTOS LEGALES:\n" + "\n".join(
        f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto'][:MAX_TEXT_CHARS]}"
        for doc in context_docs[:MAX_DOCS_TO_OPENAI]
    )

    response = openai_client.chat.completions.create(
        model=CONFIG["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{question}\n\n{context_text}"}
        ],
        temperature=CONFIG["TEMPERATURE"],
        max_tokens=CONFIG["MAX_TOKENS"]
    )

    return response.choices[0].message.content, response.usage.total_tokens


# ======================= procesar_archivo =======================
@app.route("/procesar-archivo", methods=["POST"])
def procesar_archivo():
    archivo = request.files.get("archivo")
    if not archivo:
        return jsonify({"error": "No se recibió ningún archivo"}), 400

    filename = f"/tmp/{uuid.uuid4()}_{archivo.filename}"
    archivo.save(filename)

    reader = SimpleDirectoryReader(input_files=[filename])
    documentos = reader.load_data()

    splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(documentos[0].text)

    local_index = VectorStoreIndex.from_documents(documentos)
    query_engine = local_index.as_query_engine()

    resumen_resp = query_engine.query("Resume el documento en un solo párrafo.")
    resumen = str(resumen_resp)

    document_id = str(uuid.uuid4())
    document_store[document_id] = {
        "resumen": resumen,
        "chunks": chunks,
        "query_engine": query_engine
    }

    return jsonify({ "document_id": document_id })


# ======================= preguntar =======================

@app.route("/preguntar", methods=["POST"])
def preguntar():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()
        document_id = data.get("document_id", "").strip()

        if not pregunta or not document_id:
            return jsonify({"error": "Faltan datos (pregunta o document_id)"}), 400

        doc_data = document_store.get(document_id)
        if not doc_data:
            return jsonify({"error": "document_id no válido o expirado"}), 404

        tipo = clasificar_si_es_resumen(pregunta)
        contexto = doc_data["resumen"] if tipo == "RESUMEN" else doc_data["query_engine"].query(pregunta)

        query_engine_pinecone = index.as_query_engine(similarity_top_k=TOP_K_RESULTS)
        resultado_pinecone = query_engine_pinecone.query(str(contexto))

        articulos = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}
        total_docs = len(resultado_pinecone.source_nodes)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        for i, nodo in enumerate(resultado_pinecone.source_nodes):
            if hasattr(nodo, "score") and nodo.score < 0.5:
                continue
            meta = getattr(nodo.node, 'metadata', {})
            codigo = meta.get("code", "")
            articulo = meta.get("article", "")
            texto = getattr(nodo.node, 'text', '') or meta.get("text", '')
            texto = texto.strip()

            doc = {"codigo": codigo, "articulo": articulo, "texto": texto}
            articulos.append(f"{codigo} – Art. {articulo}: {texto[:400]}")

            if i < alta_limite:
                biografia_juridica["alta"].append(doc)
            elif i < media_limite:
                biografia_juridica["media"].append(doc)
            else:
                biografia_juridica["baja"].append(doc)

        prompt_final = f"""
DOCUMENTO:
{str(contexto)}

ARTÍCULOS LEGALES RELACIONADOS:
{chr(10).join(articulos)}

PREGUNTA:
{pregunta}

Responde como abogado ecuatoriano, con base solo en el contenido anterior.
"""

        respuesta_final = openai_client.chat.completions.create(
            model=CONFIG["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": "Eres un abogado ecuatoriano experto."},
                {"role": "user", "content": prompt_final}
            ],
            temperature=CONFIG["TEMPERATURE"],
            max_tokens=CONFIG["MAX_TOKENS"]
        )

        respuesta_text = respuesta_final.choices[0].message.content.strip()
        tokens_usados = respuesta_final.usage.total_tokens

        return jsonify({
            "respuesta": respuesta_text,
            "biografia_juridica": biografia_juridica,
            "tokens_usados": { "total_tokens": tokens_usados }
        })

    except Exception as e:
        return jsonify({ "error": str(e), "traceback": traceback.format_exc() }), 500

def clasificar_si_es_resumen(pregunta):
    prompt = f"""
¿La siguiente pregunta busca un resumen general del documento o una parte específica?
Responde SOLO con una palabra: RESUMEN o CHUNKS.

Pregunta: {pregunta}
"""
    respuesta = openai_client.chat.completions.create(
        model=CONFIG["OPENAI_MODEL"],
        messages=[{ "role": "user", "content": prompt }],
        max_tokens=10,
        temperature=0
    )
    return respuesta.choices[0].message.content.strip().upper()

# ======================= MAIN ENDPOINT =======================
@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Se requiere 'question'"}), 400

        if any(kw in question.lower() for kw in MATH_KEYWORDS):
            return jsonify({"respuesta": "Lamento no realizar cálculos. Contacte al administrador."})

        match = re.search(r"art[ií]culo\s+(\d+)\s+del\s+c[oó]digo\s+([\w\s]+)", question.lower())
        articulo_buscado = match.group(1) if match else None
        codigo_buscado = normalizar(match.group(2)) if match else None

        query_engine = index.as_query_engine(similarity_top_k=TOP_K_RESULTS)
        pinecone_response = query_engine.query(question)

        context_docs = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}

        total_docs = len(pinecone_response.source_nodes)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        for i, node in enumerate(pinecone_response.source_nodes):
            score = getattr(node, 'score', 0.0)
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

        respuesta, tokens_usados = generate_legal_response(question, context_docs)

        return jsonify({
            "respuesta": respuesta,
            "biografia_juridica": biografia_juridica,
            "tokens_usados": {"total_tokens": tokens_usados}
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ======================= CALCULOS LEGALES =======================
@app.route("/calculo-legal", methods=["POST"])
def calculo_legal():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()

        if not pregunta:
            return jsonify({"error": "La pregunta es obligatoria"}), 400

        query_engine = index.as_query_engine(similarity_top_k=TOP_K_RESULTS)
        resultado_pinecone = query_engine.query(pregunta)

        context_docs = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}
        total_docs = len(resultado_pinecone.source_nodes)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        for i, nodo in enumerate(resultado_pinecone.source_nodes):
            meta = getattr(nodo.node, 'metadata', {})
            codigo = meta.get("code", "")
            articulo = meta.get("article", "")
            texto = getattr(nodo.node, 'text', '') or meta.get("text", '')
            texto = texto.strip()

            doc = {"codigo": codigo, "articulo": articulo, "texto": texto}
            context_docs.append(doc)

            if i < alta_limite:
                biografia_juridica["alta"].append(doc)
            elif i < media_limite:
                biografia_juridica["media"].append(doc)
            else:
                biografia_juridica["baja"].append(doc)

        if not context_docs:
            return jsonify({"respuesta": "No encontré normativa aplicable. No me baso en ningún artículo."})

        system_prompt = """
Eres un abogado ecuatoriano experto en cálculos legales. Aunque no has sido explícitamente entrenado para este cálculo, harás lo mejor posible para explicar detalladamente cómo se realiza, sin usar cifras numéricas bajo ninguna circunstancia en la primera parte.

---

⚖️ Parte 1: Explicación legal simbólica (siempre obligatoria)
- Usa un tono profesional, claro y didáctico.
- Explica el procedimiento de forma simbólica, utilizando únicamente nombres de variables o conceptos legales (por ejemplo: “SBU”, “remuneración mensual”, “años de servicio”).
- Detalla las fórmulas, los pasos y los artículos legales aplicables, pero evita por completo cualquier número.
- Presenta una tabla resumen simbólica (sin valores numéricos).
- Enumera al final los artículos legales en los que se fundamenta el cálculo.

⚠️ Reglas estrictas:
- Nunca utilices valores numéricos en esta sección, incluso si el usuario los proporciona.
- No resuelvas las operaciones, solo muestra la fórmula simbólica (ej.: Bonificación = SBU / 12).
- Usa términos legales exactos según la norma, como “Remuneración Básica Unificada”.

---

🧮 Parte 2: Construcción de fórmula tentativa con sustitución parcial
Solo si el usuario proporcionó suficientes datos explícitos, realiza lo siguiente:

1. **Lista de variables**:
   Enumera las variables utilizadas en la fórmula. Para cada una, indica:
   - Nombre corto (ej.: `A`, `RM`, `EDAD`)
   - Descripción clara (ej.: A = años de servicio)

2. **Fórmula general simbólica**:
   Escribe la fórmula completa usando los nombres cortos definidos.

3. **Fórmula con datos del usuario**:
   Reemplaza en la fórmula únicamente los valores mencionados explícitamente por el usuario. Los demás deben permanecer como variables simbólicas.

Ejemplo esperado:

**1. Variables**
- `A` = años de servicio
- `RM` = remuneración mensual
- `C` = coeficiente legal según edad

**2. Fórmula simbólica**
`Resultado = A × RM × C`

**3. Fórmula con datos del usuario**
`Resultado = 25 × 2000 × C`

---

⚠️ Reglas finales:
- No inventes datos faltantes.
- No resuelvas operaciones ni calcules resultados.
- Si no hay suficientes datos para hacer la fórmula, indícalo de forma clara: “No se puede construir la fórmula porque faltan variables clave como...”
"""



        context_text = "\nDOCUMENTOS LEGALES:\n" + "\n".join(
            f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto'][:MAX_TEXT_CHARS]}"
            for doc in context_docs[:MAX_DOCS_TO_OPENAI]
        )

        response = openai_client.chat.completions.create(
            model=CONFIG["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{pregunta}\n\n{context_text}"}
            ],
            temperature=CONFIG["TEMPERATURE"],
            max_tokens=CONFIG["MAX_TOKENS"]
        )

        respuesta_text = response.choices[0].message.content.strip()
        tokens_usados = response.usage.total_tokens

        return jsonify({
            "respuesta": respuesta_text,
            "biografia_juridica": biografia_juridica,
            "tokens_usados": {"total_tokens": tokens_usados}
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ======================= GENERAR CONTRATO COMPLETO =======================
@app.route("/generar-contrato-completo", methods=["POST"])
def generar_contrato_completo():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()

        if not pregunta:
            return jsonify({"error": "La pregunta es obligatoria"}), 400

        # Consulta artículos desde Pinecone
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

        return jsonify({
            "respuesta": texto,
            "tokens_usados": { "total_tokens": tokens },
            "biografia_juridica": biografia_juridica
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ======================= GENERAR CONTRATO ENDPOINT =======================
@app.route("/generar-contrato-en-partes", methods=["POST"])
def generar_contrato_en_partes():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()
        partes = int(data.get("partes", 4))  # por defecto 4 partes

        if not pregunta:
            return jsonify({"error": "La pregunta es obligatoria"}), 400

        respuestas = []
        total_tokens = 0

        for i in range(partes):
            sub_prompt = f"{pregunta}\n\nParte {i+1}. Continúa desde donde se quedó."

            r = openai_client.chat.completions.create(
                model=CONFIG["OPENAI_MODEL"],
                messages=[
                    {"role": "system", "content": "Eres un abogado ecuatoriano experto en redactar documentos legales como contratos y reglamentos extensos."},
                    {"role": "user", "content": sub_prompt}
                ],
                temperature=CONFIG["TEMPERATURE"],
                max_tokens=CONFIG["MAX_TOKENS"]
            )

            respuestas.append(r.choices[0].message.content.strip())
            total_tokens += r.usage.total_tokens

        return jsonify({
            "respuesta": "\n\n".join(respuestas),
            "biografia_juridica": { "alta": [], "media": [], "baja": [] },
            "tokens_usados": { "total_tokens": total_tokens }
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
