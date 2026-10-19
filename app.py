
import streamlit as st
import requests

from medbot_rag_answer import (
    load_retrieval_components,
    load_medbot_model,
    retrieve_evidence,
    generate_answer,
    make_history_aware_query,
)

# --------------------------------------------------
# BMI Calculator Tool
# --------------------------------------------------

def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """Calculate BMI from weight in kg and height in cm."""

    if weight_kg <= 0 or height_cm <= 0:
        return {
            "success": False,
            "message": "Weight and height must be greater than zero."
        }

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obesity"

    return {
        "success": True,
        "bmi": round(bmi, 1),
        "category": category,
        "tool": "BMI Calculator"
    }


# --------------------------------------------------
# Drug Information Tool - FDA API
# --------------------------------------------------

def get_drug_information(drug_name: str) -> dict:
    """Retrieve available drug-label information from the US FDA API."""

    if not drug_name.strip():
        return {
            "success": False,
            "message": "Please enter a medicine name."
        }

    url = "https://api.fda.gov/drug/label.json"

    params = {
        "search": (
            f'openfda.brand_name:"{drug_name}" OR '
            f'openfda.generic_name:"{drug_name}"'
        ),
        "limit": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        if response.status_code == 404:
            return {
                "success": False,
                "message": (
                    f"No FDA drug-label information was found for "
                    f"'{drug_name}'. Try another medicine name."
                )
            }

        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])

        if not results:
            return {
                "success": False,
                "message": "No drug information was returned by the FDA."
            }

        record = results[0]
        openfda = record.get("openfda", {})

        def get_first_text(field):
            value = record.get(field, [])
            return value[0] if value else "Not available in this label."

        return {
            "success": True,
            "tool": "DrugInfo - US FDA API",
            "drug_name": (
                (openfda.get("brand_name") or
                 openfda.get("generic_name") or
                 [drug_name])[0]
            ),
            "purpose": get_first_text("purpose"),
            "indications": get_first_text("indications_and_usage"),
            "warnings": get_first_text("warnings"),
            "source": "https://api.fda.gov/drug/label.json"
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": "The FDA API request timed out. Please try again."
        }

    except requests.exceptions.RequestException:
        return {
            "success": False,
            "message": "Unable to connect to the FDA API. Please try again."
        }

    except (ValueError, KeyError, IndexError):
        return {
            "success": False,
            "message": "The FDA response could not be processed."
        }


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="MedBot | Medical Q&A Assistant",
    page_icon="🩺",
    layout="wide",
)

# --------------------------------------------------
# Page styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #167D9A;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
    }
    .disclaimer {
        background-color: #fff4df;
        padding: 14px;
        border-radius: 8px;
        border-left: 5px solid #e5a000;
        color: #594016;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🩺 MedBot</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'AI-Powered Medical Question & Answer Assistant'
    '</div>',
    unsafe_allow_html=True,
)

st.write("Ask a medical question and explore the evidence "
         "retrieved from trusted medical sources.")

st.markdown(
    """
    <div class="disclaimer">
    <strong>Medical disclaimer:</strong> MedBot provides
    informational support only. It is not a substitute for
    professional medical advice, diagnosis, or treatment.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# BMI Calculator Tool
# --------------------------------------------------
st.divider()
st.subheader("🧮 Medical Tool: BMI Calculator")
st.write("Calculate Body Mass Index (BMI) using height and weight.")

with st.form("bmi_calculator_form"):
    col1, col2 = st.columns(2)

    with col1:
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=1.0,
            max_value=500.0,
            value=70.0,
            step=0.5,
        )

    with col2:
        height_cm = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=170.0,
            step=0.5,
        )

    calculate_button = st.form_submit_button(
        "Calculate BMI",
        use_container_width=True,
    )

if calculate_button:
    result = calculate_bmi(weight_kg, height_cm)

    st.markdown("#### Tool-call trace")
    st.code(
        "Tool: BMI Calculator\n"
        f"Input: weight={weight_kg} kg, height={height_cm} cm\n"
        "Status: Executed locally"
    )

    if result["success"]:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Your BMI", result["bmi"])

        with col2:
            st.metric("BMI category", result["category"])

        st.caption(
            "BMI is a screening measure, not a diagnosis. "
            "Interpretation can vary by age and other factors."
        )
    else:
        st.error(result["message"])


# --------------------------------------------------
# Drug Information Tool
# --------------------------------------------------

st.divider()
st.subheader("💊 Medical Tool: Drug Information")
st.write(
    "Search available medicine-label information from the US FDA."
)

with st.form("drug_information_form"):
    drug_name = st.text_input(
        "Medicine name",
        placeholder="Example: ibuprofen"
    )

    drug_button = st.form_submit_button(
        "Search Drug Information",
        use_container_width=True
    )

if drug_button:
    with st.spinner("Searching the FDA drug-label database..."):
        result = get_drug_information(drug_name)

    st.markdown("#### Tool-call trace")
    st.code(
        "Tool: DrugInfo - US FDA API\n"
        f"Input: medicine={drug_name}\n"
        f"Status: {'Completed' if result['success'] else 'Not completed'}"
    )

    if result["success"]:
        st.markdown("#### Drug information")
        st.write(f"**Medicine:** {result['drug_name']}")

        st.markdown("**Purpose**")
        st.write(result["purpose"])

        st.markdown("**Indications and usage**")
        st.write(result["indications"])

        st.markdown("**Warnings**")
        st.write(result["warnings"])

        st.markdown(
            f"[Open FDA drug-label API]({result['source']})"
        )

        st.caption(
            "Drug labels may be incomplete or vary by product. "
            "This information is not a dosage recommendation. "
            "Consult a qualified healthcare professional."
        )
    else:
        st.warning(result["message"])

# --------------------------------------------------
# Load components only once per session cache
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_components():
    index, metadata, embedding_model = load_retrieval_components()
    tokenizer, model = load_medbot_model()

    return (
        index,
        metadata,
        embedding_model,
        tokenizer,
        model,
    )

if "medbot_chat_history" not in st.session_state:
    st.session_state.medbot_chat_history = []
# --------------------------------------------------
# User interface
# --------------------------------------------------
st.subheader("Ask MedBot")

question = st.text_area(
    "Enter your medical question",
    placeholder="Example: What are the symptoms of diabetes?",
    height=100,
)

ask_button = st.button(
    "Get Medical Answer",
    type="primary",
    use_container_width=True,
)

if ask_button:
    if not question.strip():
        st.warning("Please enter a medical question.")
    else:
        try:
            with st.spinner(
                "Loading MedBot and retrieving medical evidence..."
            ):
                (
                    index,
                    metadata,
                    embedding_model,
                    tokenizer,
                    model,
                ) = load_components()

            
            with st.spinner("Searching medical evidence..."):
                retrieval_question = make_history_aware_query(
                    question.strip(),
                    st.session_state.medbot_chat_history,
                )
                st.caption(f"FAISS retrieval query: {retrieval_question}")
                evidence = retrieve_evidence(
                    retrieval_question,
                    index,
                    metadata,
                    embedding_model,
                )


            
            with st.spinner("Generating your answer..."):
                answer = generate_answer(
                    retrieval_question,
                    evidence,
                    tokenizer,
                    model,
                )

            st.session_state.medbot_chat_history.extend([
                {"role": "user", "content": question.strip()},
                {"role": "assistant", "content": answer},
            ])

            st.session_state.medbot_chat_history = (
                st.session_state.medbot_chat_history[-12:]
            )


            st.subheader("MedBot Answer")
            st.markdown(answer)

            st.divider()
            st.subheader("Retrieved Medical Evidence")

            
            if not evidence:
                st.info(
                    "No evidence passages were retrieved for this question."
                )
            else:
                
                # Keep the highest-ranked passage for each source URL.
                # Hide very low-similarity passages in the UI.
                # Filter evidence for display using the question's medical topic.
                question_lower = question.lower()

                medical_topics = [
                    "diabetes",
                    "asthma",
                    "high blood pressure",
                    "hypertension",
                    "migraine",
                    "arthritis",
                    "heart disease",
                ]

                requested_topics = [
                    topic for topic in medical_topics
                    if topic in question_lower
                ]

                display_evidence = [
                    item for item in evidence
                    if float(item.get("score", 0.0)) >= 0.30
                    and (
                        not requested_topics
                        or any(
                            topic in (
                                str(item.get("title", "")) + " "
                                + str(item.get("text", item.get("passage", "")))
                            ).lower()
                            for topic in requested_topics
                        )
                    )
                ]
                if not display_evidence:
                    st.info(
                        "No sufficiently relevant medical evidence was found "
                        "for this question. Try rephrasing your question or "
                        "consult a trusted medical source."
                    )
                unique_evidence = []
                seen_urls = set()

                for item in display_evidence:
                    url = item.get("url", "").strip()

                    if url:
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)

                    unique_evidence.append(item)

                st.caption(
                    f"Showing {len(unique_evidence)} unique source(s) "
                    f"from {len(display_evidence)} displayed passage(s)."
                )

                for number, item in enumerate(unique_evidence, start=1):
                    title = item.get("title", "Medical source")
                    organization = item.get(
                        "organization",
                        item.get("source", "Not specified"),
                    )
                    url = item.get("url", "")
                    text = item.get("text", "")
                    score = item.get("score")

                    with st.container(border=True):
                        st.markdown(f"**[{number}] {title}**")

                        st.caption(f"Organization: {organization}")

                        
                        if score is not None:
                            score_value = float(score)
                            st.caption(f"Similarity score: {score_value:.4f}")

                            if score_value >= 0.70:
                                st.success("High text similarity")
                            elif score_value >= 0.50:
                                st.info("Moderate text similarity")
                            else:
                                st.caption("Low text similarity")


                        
                        preview_length = 220

                        if len(text) > preview_length:
                            st.caption(text[:preview_length].rstrip() + "...")
                        else:
                            st.caption(text)

                        with st.expander("View full retrieved passage"):
                            st.write(text)


                        if url:
                            st.markdown(f"[Open original source ↗]({url})")



            st.caption(
                "Always verify important health decisions with "
                "a qualified healthcare professional."
            )

        except Exception as error:
            st.error(
                "MedBot encountered an error while processing "
                "your question."
            )
            st.exception(error)
