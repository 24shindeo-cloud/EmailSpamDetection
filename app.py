
import pickle
import gradio as gr
from pathlib import Path

# ---------------- LOAD TRAINED MODEL ----------------

MODEL_PATH = Path(__file__).with_name("best_spam_email_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ---------------- EMAIL PREDICTION ----------------

def analyze_email(message):
    if not message or not message.strip():
        return (
            "⏳ **Waiting for email**",
            "Enter an email message to begin analysis."
        )

    try:
        prediction = str(
            model.predict([message.strip()])[0]
        ).lower()

        if prediction == "spam":
            return (
                "🚨 **SPAM DETECTED**",
                """
                This message may be suspicious.

                - Avoid clicking unknown links.
                - Do not share passwords or OTPs.
                - Verify the sender before responding.
                """
            )

        return (
            "✅ **NOT SPAM**",
            """
            The model classified this message as legitimate.

            Always verify unexpected requests and links,
            even when a message is classified as not spam.
            """
        )

    except Exception as error:
        return (
            "⚠️ **Analysis failed**",
            f"Unable to analyze this message: {error}"
        )


# ---------------- CLEAR INPUT ----------------

def clear_form():
    return (
        "",
        "⏳ **Waiting for email**",
        "Enter an email message to begin analysis."
    )


# ---------------- CUSTOM DESIGN ----------------

css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
    background: #0b1120 !important;
}

#header {
    background: linear-gradient(120deg, #172554, #164e63);
    padding: 32px 20px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 22px;
    border: 1px solid #334155;
}

#header h1 {
    font-size: 36px;
    margin-bottom: 8px;
}

#header p {
    color: #cbd5e1;
    font-size: 15px;
}

.panel {
    border: 1px solid #334155 !important;
    border-radius: 18px !important;
    padding: 22px !important;
    background: #111c30 !important;
}

#analyze {
    background: linear-gradient(90deg, #2563eb, #0891b2) !important;
    color: white !important;
    border: none !important;
}

#result {
    border-radius: 14px;
    padding: 16px;
    background: #0b1120;
    border: 1px solid #334155;
    min-height: 55px;
}

footer {
    display: none !important;
}
"""


# ---------------- BUILD DASHBOARD ----------------

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=css,
    title="SpamShield AI"
) as demo:

    gr.HTML("""
    <div id="header">
        <h1>🛡️ SpamShield AI</h1>
        <p>Intelligent Email Security & Spam Classification Dashboard</p>
        <p>Machine Learning · Text Analysis · Email Protection</p>
    </div>
    """)

    with gr.Row():

        # LEFT PANEL: EMAIL INPUT
        with gr.Column(
            scale=6,
            elem_classes=["panel"]
        ):
            gr.Markdown("## ✉️ Email Analyzer")

            gr.Markdown(
                "Paste the email text below to check its classification."
            )

            email = gr.Textbox(
                label="Email content",
                placeholder="Paste your email subject and message here...",
                lines=10
            )

            with gr.Row():
                analyze = gr.Button(
                    "🔍 Analyze Email",
                    variant="primary",
                    elem_id="analyze"
                )

                clear = gr.Button("↻ Clear")

            gr.Markdown("### Try a sample email")

            gr.Examples(
                examples=[
                    [
                        "Congratulations! You won a free prize. "
                        "Click here now to claim your reward!"
                    ],
                    [
                        "Hi, our project meeting is scheduled "
                        "for tomorrow at 10 AM. See you there."
                    ],
                    [
                        "URGENT! Claim your cash reward now. "
                        "Limited time offer!"
                    ]
                ],
                inputs=email
            )

        # RIGHT PANEL: RESULT
        with gr.Column(
            scale=5,
            elem_classes=["panel"]
        ):
            gr.Markdown("## 📊 Analysis Results")

            result = gr.Markdown(
                "⏳ **Waiting for email**",
                elem_id="result"
            )

            details = gr.Markdown(
                "Enter an email message to begin analysis."
            )

            gr.Markdown("""
            ### 🔐 Security Tips

            - Verify the sender's email address.
            - Avoid unexpected links and attachments.
            - Never share passwords or OTPs by email.

            *Predictions are estimates, not a guarantee of safety.*
            """)

    # BUTTON ACTIONS
    analyze.click(
        fn=analyze_email,
        inputs=email,
        outputs=[result, details]
    )

    clear.click(
        fn=clear_form,
        inputs=[],
        outputs=[email, result, details]
    )


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    demo.launch()
