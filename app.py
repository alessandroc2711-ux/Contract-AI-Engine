import os
import re
import json
import streamlit as st
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from docx import Document
from openai import OpenAI

# =========================
# INIT CLIENT
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# SESSION STATE
# =========================
def init_state():
    defaults = {
        "template_contract": None,
        "example_contract": None,
        "example_request": None,
        "example_event": None,
        "new_request": None,
        "new_event": None,
        "mapping": None,
        "extracted_fields": None,

        "debug_template_fields": None,
        "debug_mapping_raw_fields": None,
        "debug_mapping_coverage": None,
        "debug_doc_scan": None,

        "debug_extraction_input": None,
        "debug_extraction_prompt": None,
        "debug_extraction_raw_output": None,
        "debug_extraction_parsed": None,
        "debug_extraction_before_filter": None,
        "debug_extraction_after_filter": None,

        "final_docx": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def require(key, msg):
    if not st.session_state.get(key):
        st.error(msg)
        st.stop()


init_state()

# =========================
# AI ENGINE
# =========================
class AIEngine:

    def safe_json_loads(self, text):
        try:
            return json.loads(text)
        except Exception:
            text = str(text).strip()
            text = re.sub(r"^```json", "", text)
            text = re.sub(r"^```", "", text)
            text = re.sub(r"```$", "", text)
            text = text.strip()
            text = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', text)

            try:
                return json.loads(text)
            except Exception:
                return {"error": "invalid_json", "raw": text}

    def build_mapping(
        self,
        template_fields,
        example_contract,
        example_request,
        example_event
    ):

        prompt = f"""
Return ONLY JSON.

You are building a FIELD MAPPING SYSTEM.

You MUST learn from examples.

=== TEMPLATE FIELDS ===
{template_fields}

=== EXAMPLE CONTRACT (structure reference) ===
{example_contract}

=== EXAMPLE REQUEST (meaning reference) ===
{example_request}

=== EXAMPLE EVENT (meaning reference) ===
{example_event}

TASK:
For each template field:
- understand semantic meaning from examples
- decide where value comes from (REQUEST or EVENT)
- define extraction rule if needed

OUTPUT FORMAT:
{{
  "fields": [
    {{
      "field_name": "...",
      "source": "REQUEST|EVENT",
      "semantic_meaning": "...",
      "rule": "optional"
    }}
  ]
}}
"""

        res = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        result = self.safe_json_loads(res.choices[0].message.content)
        st.session_state["debug_mapping_raw_fields"] = result
        return result

    def extract_fields(self, mapping, new_request, new_event):

        allowed_fields = [f["field_name"] for f in mapping.get("fields", [])]

        request_text = new_request or ""
        event_text = new_event or ""

        st.session_state["debug_extraction_input"] = {
            "allowed_fields": allowed_fields,
            "request_len": len(request_text),
            "event_len": len(event_text)
        }

        prompt = f"""
Return ONLY JSON.

You are doing STRICT EXTRACTION.

RULES:
- Use ONLY REQUEST or EVENT text
- Every value MUST be supported by evidence text
- If uncertain, return empty value

KNOWN FIELD SEMANTICS:
{json.dumps(mapping.get("fields", []), indent=2)}

REQUEST:
{request_text}

EVENT:
{event_text}

OUTPUT FORMAT:
{{
  "fields": [
    {{
      "field_name": "...",
      "value": "...",
      "evidence": "...",
      "source": "REQUEST|EVENT"
    }}
  ]
}}
"""

        st.session_state["debug_extraction_prompt"] = prompt

        res = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        raw = res.choices[0].message.content
        st.session_state["debug_extraction_raw_output"] = raw

        result = self.safe_json_loads(raw)
        st.session_state["debug_extraction_parsed"] = result

        if not isinstance(result, dict):
            result = {"fields": []}

        result.setdefault("fields", [])

        st.session_state["debug_extraction_before_filter"] = result["fields"]

        filtered = [
            f for f in result["fields"]
            if f.get("field_name") in allowed_fields
        ]

        result["fields"] = filtered
        st.session_state["debug_extraction_after_filter"] = filtered

        return result


# =========================
# DOCUMENT PARSER (ONLY FIX HERE)
# =========================
class DocumentParser:

    def parse(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".pdf":
            return self._pdf(path)
        if ext == ".docx":
            return self._docx(path)
        return self._txt(path)

    def _pdf(self, path):
        text_layer = self._text_layer(path)
        ocr_text = self._ocr(path)
        return {"text": self._merge(text_layer, ocr_text)}

    # =========================
    # FIX: ROBUST FOOTER EXTRACTION (DEPLOY SAFE)
    # =========================
    def _text_layer(self, path):
        out = []

        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:

                t = None

                # 1) 🔥 most robust in cloud: word-based extraction
                words = p.extract_words()
                if words:
                    t = " ".join(w.get("text", "") for w in words)

                # 2) standard extraction
                if not t:
                    t = p.extract_text()

                # 3) layout-aware fallback
                if not t:
                    t = p.extract_text(layout=True)

                # 4) tolerance fallback (fix shifted footer/header)
                if not t:
                    t = p.extract_text(x_tolerance=3, y_tolerance=3)

                # 5) full-page safety crop fallback
                if not t:
                    cropped = p.crop((0, 0, p.width, p.height))
                    words = cropped.extract_words()
                    if words:
                        t = " ".join(w.get("text", "") for w in words)

                if t:
                    out.append(t)

        return "\n".join(out)

    def _ocr(self, path):
        try:
            imgs = convert_from_path(path, dpi=300)
        except Exception:
            return ""

        return "\n".join(pytesseract.image_to_string(i) for i in imgs)

    def _merge(self, a, b):
        return "\n".join(dict.fromkeys((a + "\n" + b).splitlines()))

    def _docx(self, path):
        doc = Document(path)

        parts = []
        for p in doc.paragraphs:
            parts.append("".join(run.text for run in p.runs))

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        parts.append("".join(run.text for run in p.runs))

        text = "\n".join(parts)

        st.session_state["debug_doc_scan"] = {
            "paragraphs": len(doc.paragraphs),
            "tables": len(doc.tables),
            "chars": len(text)
        }

        return {"text": text}

    def _txt(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return {"text": f.read()}


parser = DocumentParser()
ai = AIEngine()

# =========================
# HELPERS
# =========================
def upload(label, key, state):
    f = st.file_uploader(label, key=key)
    if f:
        path = os.path.join("temp", f.name)
        with open(path, "wb") as w:
            w.write(f.read())
        st.session_state[state] = path


def parse_text(path):
    return parser.parse(path)["text"]


def extract_template_fields(text):
    raw = re.findall(r"\{\{\s*([^}]+?)\s*\}\}", text)

    st.session_state["debug_template_fields"] = {
        "raw_fields": raw,
        "found": len(raw),
        "unique": len(set(raw))
    }

    return sorted(set(raw))


def replace_docx(doc, repl):
    safe_map = {f"{{{{{k}}}}}": str(v) for k, v in repl.items()}

    def replace_in_paragraph(paragraph):
        full_text = "".join(run.text for run in paragraph.runs)
        original = full_text

        for k, v in safe_map.items():
            full_text = full_text.replace(k, v)

        if full_text != original and paragraph.runs:
            paragraph.runs[0].text = full_text
            for r in paragraph.runs[1:]:
                r.text = ""

    for p in doc.paragraphs:
        replace_in_paragraph(p)

    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    replace_in_paragraph(p)


# =========================
# UI
# =========================
st.set_page_config(page_title="Contract AI", layout="wide")
st.title("📄 Contract AI Engine (Single Page)")

os.makedirs("temp", exist_ok=True)

st.header("1️⃣ Upload documents")

col1, col2 = st.columns(2)

with col1:
    upload("Template Contract", "t", "template_contract")
    upload("Example Contract", "ec", "example_contract")

with col2:
    upload("Example Request", "er", "example_request")
    upload("Example Event", "ee", "example_event")

upload("New Request", "nr", "new_request")
upload("New Event", "ne", "new_event")

st.header("2️⃣ Mapping")

if st.button("Run Mapping"):
    require("template_contract", "Missing template contract")

    tpl = parse_text(st.session_state["template_contract"])
    fields = extract_template_fields(tpl)

    m = ai.build_mapping(
        fields,
        parse_text(st.session_state.get("example_contract") or ""),
        parse_text(st.session_state.get("example_request") or ""),
        parse_text(st.session_state.get("example_event") or "")
    )

    st.session_state["mapping"] = m


st.subheader("Template fields detected")

if st.session_state.get("debug_template_fields"):
    st.json(st.session_state["debug_template_fields"])
else:
    st.info("Run mapping to detect template placeholders.")

st.subheader("Mapping result (FULL)")

if st.session_state.get("mapping"):
    st.json(st.session_state["mapping"])
else:
    st.info("No mapping generated yet.")

if st.session_state.get("mapping"):
    st.subheader("Mapped fields (readable view)")
    st.table([
        {
            "field": f.get("field_name"),
            "source": f.get("source"),
            "rule": f.get("rule")
        }
        for f in st.session_state["mapping"].get("fields", [])
    ])

st.header("3️⃣ Extraction")

if st.button("Run Extraction"):
    require("mapping", "Run mapping first")

    ex = ai.extract_fields(
        st.session_state["mapping"],
        parse_text(st.session_state.get("new_request") or ""),
        parse_text(st.session_state.get("new_event") or "")
    )

    st.session_state["extracted_fields"] = ex


st.subheader("Extracted values (LIVE)")

if st.session_state.get("extracted_fields"):
    st.json(st.session_state["extracted_fields"])
else:
    st.info("Run extraction to see extracted values.")

st.header("4️⃣ Generate Contract")

if st.button("Generate"):
    require("extracted_fields", "No extracted data")

    doc = Document(st.session_state["template_contract"])

    repl = {
        f["field_name"]: f["value"]
        for f in st.session_state["extracted_fields"].get("fields", [])
    }

    st.json(repl)

    replace_docx(doc, repl)

    out = "temp/out.docx"
    doc.save(out)

    st.session_state["final_docx"] = out
    st.success("Contract generated")

if st.session_state.get("final_docx"):
    with open(st.session_state["final_docx"], "rb") as f:
        st.download_button("Download contract", f, file_name="contract.docx")
