import os
from pathlib import Path
import html
import traceback

import gradio as gr
from generation.llm import GeminiServiceError
from generation.pipeline import LegalQAPipeline


APP_TITLE = "Lexis Lab"

# Two tones only: ink (#0a0a0a) and bone (#e9e4d8). Everything else is bone at varying opacity.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500&display=swap');

:root {
    --ink: #0a0a0a;
    --bone: #e9e4d8;
    --bone-90: rgba(233,228,216,.90);
    --bone-60: rgba(233,228,216,.60);
    --bone-40: rgba(233,228,216,.40);
    --bone-12: rgba(233,228,216,.12);
    --bone-06: rgba(233,228,216,.06);
    --bone-03: rgba(233,228,216,.03);
    --serif: 'Instrument Serif', 'Times New Roman', serif;
    --sans: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}

/* full-bleed black: only html/body paint; the container is transparent so there is no boxed column */
html, body, gradio-app {
    background: var(--ink) !important;
    color: var(--bone-90) !important;
    font-family: var(--sans) !important;
    min-height: 100vh;
}
.gradio-container, .gradio-container .main, .gradio-container .wrap {
    background: transparent !important;
}

/* cinematic light + film grain */
body::before {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(60vw 50vh at 85% -5%, rgba(233,228,216,.16), transparent 70%),
        radial-gradient(50vw 60vh at -5% 105%, rgba(233,228,216,.09), transparent 70%);
}
body::after {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: .07;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .9 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
}

.gradio-container {
    max-width: 1320px !important;
    margin: auto !important;
    padding: 56px 40px 40px !important;
    position: relative; z-index: 1;
}

/* header */
#app-header { padding: 0 4px 44px; border-bottom: 1px solid var(--bone-12); margin-bottom: 36px; }
#app-header h1 {
    font-family: var(--serif);
    font-weight: 400;
    font-size: clamp(56px, 8vw, 104px);
    line-height: .95;
    letter-spacing: -0.02em;
    margin: 0 0 18px;
    color: var(--bone);
    text-shadow: 0 0 60px rgba(233,228,216,.18);
}
#app-header p {
    margin: 0; max-width: 46ch;
    font-size: 14px; font-weight: 300; line-height: 1.6;
    color: var(--bone-60);
}

/* glass panels: one card per column, nothing nested inside has its own box */
.glass {
    background: linear-gradient(145deg, var(--bone-06), var(--bone-03)) !important;
    border: 1px solid var(--bone-12) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    backdrop-filter: blur(24px) saturate(120%);
    -webkit-backdrop-filter: blur(24px) saturate(120%);
    box-shadow: inset 0 1px 0 rgba(233,228,216,.10), 0 30px 80px rgba(0,0,0,.55);
    gap: 14px !important;
}
.glass .block, .glass .form, .glass .prose, .glass .gr-group {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.panel-title { font-family: var(--serif); font-size: 28px; color: var(--bone); line-height: 1.1; margin: 0; }
.panel-description { font-size: 13px; font-weight: 300; color: var(--bone-60); line-height: 1.6; margin: 6px 0 0; }

/* labels */
label span, .gradio-container label > span {
    color: var(--bone-60) !important; font-size: 12px !important; font-weight: 400 !important;
}
.glass .block > label > span:first-child, .glass [data-testid="block-label"] {
    background: transparent !important; border: none !important; box-shadow: none !important;
}

/* question: soft fill, border only on focus */
textarea, input[type=text] {
    background: var(--bone-03) !important;
    color: var(--bone) !important;
    border: 1px solid transparent !important;
    border-radius: 14px !important;
    font-family: var(--sans) !important; font-weight: 300 !important; font-size: 15px !important;
    box-shadow: none !important;
}
textarea::placeholder { color: var(--bone-40) !important; }
textarea:focus, input:focus { border-color: var(--bone-40) !important; outline: none !important; }

/* file drop: one soft dashed zone, no label chip */
[data-testid="file-upload"], .upload-container, .file-preview-holder, .file-preview {
    background: var(--bone-03) !important;
    border: 1px dashed var(--bone-12) !important;
    border-radius: 14px !important;
    color: var(--bone-60) !important;
    min-height: 0 !important;
}
[data-testid="file-upload"]:hover { border-color: var(--bone-40) !important; }

/* buttons */
#process-btn, #ask-btn {
    min-height: 46px; border-radius: 999px !important;
    font-family: var(--sans) !important; font-weight: 500 !important; letter-spacing: .01em;
    transition: background .2s, color .2s, box-shadow .2s;
}
#ask-btn {
    background: var(--bone) !important; color: var(--ink) !important; border: 1px solid var(--bone) !important;
}
#ask-btn:hover { box-shadow: 0 0 40px rgba(233,228,216,.35); }
#process-btn {
    background: transparent !important; color: var(--bone) !important; border: 1px solid var(--bone-40) !important;
}
#process-btn:hover { background: var(--bone-06) !important; border-color: var(--bone) !important; }

/* sources: two columns on wide screens */
.src-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); column-gap: 40px; }

/* answer: no box, a hairline above, height follows content */
#answer-panel, #answer-panel > *, #answer-panel .block, #answer-panel .prose {
    background: transparent !important; border: none !important; box-shadow: none !important;
    min-height: 0 !important; padding: 0 !important;
}
#answer-panel { max-width: 90ch; }
#answer-panel h3 { font-family: var(--serif); font-weight: 400; font-size: 22px; color: var(--bone); margin-top: 0; }
#answer-panel p, #answer-panel li { font-size: 15px; font-weight: 300; line-height: 1.75; color: var(--bone-90); }

#status-text, #status-text p { color: var(--bone-60) !important; font-size: 13px; font-weight: 300; }
.prose strong { color: var(--bone); font-weight: 500; }
.prose code {
    background: var(--bone-06) !important; color: var(--bone) !important;
    border: 1px solid var(--bone-12); border-radius: 6px; padding: 1px 6px;
}
.prose blockquote {
    border-left: 1px solid var(--bone-40) !important; background: transparent !important;
    color: var(--bone-60) !important; font-family: var(--serif); font-size: 17px; font-style: italic; padding-left: 16px;
}
.prose hr { border-color: var(--bone-12) !important; }
.prose h3 { color: var(--bone); font-family: var(--serif); font-weight: 400; }

/* accordion: hairline, not a box */
.accordion, details {
    background: transparent !important; border: none !important;
    border-top: 1px solid var(--bone-12) !important; border-radius: 0 !important;
}
.accordion .label-wrap, summary { color: var(--bone-60) !important; font-size: 13px !important; }

.fineprint { text-align: center; font-size: 12px; font-weight: 300; line-height: 1.6; color: var(--bone-40); }
.fineprint p { color: var(--bone-40) !important; }
#footer-note { text-align: center; color: var(--bone-40); font-size: 12px; font-weight: 300; margin-top: 8px; }

footer { display: none !important; }

@media (max-width: 768px) {
    .gradio-container { padding: 32px 18px !important; }
    .glass { padding: 20px !important; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

THEME = gr.themes.Base(
    primary_hue="stone",
    secondary_hue="stone",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "Helvetica Neue", "sans-serif"],
).set(
    body_background_fill="#0a0a0a", body_background_fill_dark="#0a0a0a",
    body_text_color="#e9e4d8", body_text_color_dark="#e9e4d8",
    block_background_fill="transparent", block_background_fill_dark="transparent",
    block_border_width="0px", block_shadow="none",
    input_background_fill="rgba(233,228,216,.03)", input_background_fill_dark="rgba(233,228,216,.03)",
    input_border_color="transparent", input_border_color_dark="transparent",
    button_primary_background_fill="#e9e4d8", button_primary_background_fill_dark="#e9e4d8",
    button_primary_text_color="#0a0a0a", button_primary_text_color_dark="#0a0a0a",
)

FORCE_DARK = "() => { document.body.classList.add('dark'); }"


def _get(source, key, default=None):
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def render_sources(sources):
    """Render every source as its own item: metadata, short preview, expandable full text."""
    items = []
    for index, source in enumerate(sources, start=1):
        if isinstance(source, dict):
            fields = dict(source)
        elif hasattr(source, "model_dump"):
            fields = source.model_dump()
        else:
            fields = dict(getattr(source, "__dict__", {}))

        for key in ("page_number", "chunk_id"):
            if key not in fields and _get(source, key) is not None:
                fields[key] = _get(source, key)

        text = _get(source, "text", "") or ""
        meta = [
            (k, v) for k, v in fields.items()
            if k != "text" and not k.startswith("_")
            and isinstance(v, (str, int, float, bool)) and v != ""
        ]
        meta.sort(key=lambda kv: {"page_number": 0, "chunk_id": 1}.get(kv[0], 2))

        title = f"Source {index}"
        if _get(source, "page_number") is not None:
            title += f", page {html.escape(str(_get(source, 'page_number')))}"
        meta_html = " &nbsp; ".join(
            f"<code>{html.escape(k)}: {html.escape(str(v))}</code>"
            for k, v in meta if k != "page_number"
        )
        safe = html.escape(text)

        items.append(
            '<div style="padding:16px 0;border-bottom:1px solid var(--bone-12)">'
            f'<div style="font-family:var(--serif);font-size:20px;color:var(--bone)">{title}</div>'
            f'<div style="margin:6px 0 10px;font-size:12px">{meta_html}</div>'
            '<div style="display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;'
            'overflow:hidden;border-left:1px solid var(--bone-40);padding-left:16px;'
            f'font-family:var(--serif);font-style:italic;font-size:17px;color:var(--bone-60)">{safe}</div>'
            '<details style="margin-top:10px;border:none">'
            '<summary style="cursor:pointer;font-size:13px;color:var(--bone-60)">Show full passage</summary>'
            '<div style="white-space:pre-wrap;margin-top:10px;border-left:1px solid var(--bone-40);'
            f'padding-left:16px;font-size:14px;font-weight:300;line-height:1.7;color:var(--bone-90)">{safe}</div>'
            '</details></div>'
        )
    return '<div class="src-grid">' + "".join(items) + "</div>"


def process_document(file_path):
    """Load a PDF and build the existing retrieval pipeline."""
    if not file_path:
        return None, "Upload a PDF before processing it."

    path = Path(file_path)

    if not path.exists() or path.suffix.lower() != ".pdf":
        return None, "Please upload a valid PDF file."

    try:
        pipeline = LegalQAPipeline(pdf_path=path, document_id=path.stem)
        return (
            pipeline,
            f"**Document ready**\n\n`{path.name}` is processed. Ask a question on the right.",
        )

    except Exception as exc:
        print(traceback.format_exc())
        return None, (
            "**Could not process this document.**\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Check the terminal for the full error."
        )


def answer_question(question, pipeline):
    """Ask the existing pipeline and format its answer and sources."""
    if pipeline is None:
        return (
            "### Upload a document first\n"
            "Upload a PDF and select **Process document** before asking questions.",
            "Waiting for a document",
            "No source passages available yet.",
        )

    if not question or not question.strip():
        return (
            "Enter a question about the document.",
            "Waiting for a question",
            "No source passages available.",
        )

    try:
        result = pipeline.answer(question.strip(), top_k=5)

        answer_text = getattr(result, "answer", str(result))
        sources = getattr(result, "sources", []) or []
        insufficient = getattr(result, "insufficient_evidence", False)
        citations_valid = getattr(result, "citations_valid", False)
        confidence = getattr(result, "confidence", None)

        if insufficient:
            status = (
                "**Insufficient evidence.** The retrieved material "
                "does not support a reliable answer."
            )
        elif citations_valid:
            status = (
                "**Citations check out.** Cited page numbers match "
                "the retrieved sources. This does not independently "
                "verify every factual claim."
            )
        else:
            status = (
                "**Review citations.** The answer was generated, "
                "but its citations did not pass the current validation check."
            )

        if sources and not insufficient:
            source_markdown = render_sources(sources)
        else:
            source_markdown = "No source passages were returned."

        return answer_text, status, source_markdown

    except Exception as exc:
        print(traceback.format_exc())
        return (
            f"### Something went wrong\n`{type(exc).__name__}: {exc}`",
            "The question could not be answered. Check the terminal for details.",
            "No source passages available.",
        )


with gr.Blocks(theme=THEME, css=CSS, js=FORCE_DARK, title=APP_TITLE) as demo:

    pipeline_state = gr.State(None)

    gr.HTML(
        """
        <div id="app-header">
            <h1>Lexis Lab</h1>
            <p>Grounded answers from your legal documents, with every claim traced back to the page it came from.</p>
        </div>
        """
    )

    # top: two short cards of equal height
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=280, elem_classes="glass"):
            gr.HTML(
                """
                <div class="panel-title">Your document</div>
                <div class="panel-description">Upload a legal PDF to begin.</div>
                """
            )
            pdf_file = gr.File(
                label="Legal PDF", show_label=False,
                file_types=[".pdf"], type="filepath", height=150,
            )
            process_btn = gr.Button("Process document", elem_id="process-btn")
            document_status = gr.Markdown("Waiting for a document.", elem_id="status-text")

        with gr.Column(scale=2, min_width=400, elem_classes="glass"):
            gr.HTML(
                """
                <div class="panel-title">Ask your document</div>
                <div class="panel-description">
                    Clauses, obligations, dates, termination, confidentiality, or any other term in the PDF.
                </div>
                """
            )
            question_box = gr.Textbox(
                label="Your question",
                placeholder="What does the agreement say about confidentiality?",
                lines=5,
                max_lines=8,
            )
            ask_btn = gr.Button("Analyse question", elem_id="ask-btn")

    # bottom: full-width results, so nothing leaves an empty column beside it
    with gr.Column(elem_classes="glass", elem_id="results"):
        answer_output = gr.Markdown(
            "Your answer will appear here after you process a document.", elem_id="answer-panel"
        )
        status_output = gr.Markdown("Citation and evidence status will appear here.", elem_id="status-text")
        with gr.Accordion("Retrieved source passages", open=False):
            sources_output = gr.HTML("Source passages will appear here.")

    gr.Markdown(
        "AI-generated responses can be wrong. Verify important legal "
        "conclusions against the original document.",
        elem_classes="fineprint",
    )

    gr.HTML('<div id="footer-note">Lexis Lab. Retrieval-augmented legal document analysis.</div>')

    process_btn.click(process_document, [pdf_file], [pipeline_state, document_status])
    ask_btn.click(answer_question, [question_box, pipeline_state], [answer_output, status_output, sources_output])
    question_box.submit(answer_question, [question_box, pipeline_state], [answer_output, status_output, sources_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        )