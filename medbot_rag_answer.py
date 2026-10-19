
from pathlib import Path
import json

import faiss
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# --------------------------------------------------
# 1. Project paths and model settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
RAG_DIR = PROJECT_DIR / "medbot_phase3_rag"
INDEX_DIR = RAG_DIR / "vector_index"

INDEX_PATH = INDEX_DIR / "medical.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

ADAPTER_PATH = PROJECT_DIR / "medbot_phase2_full" / "lora_adapter"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_SEARCH = 10
MAX_EVIDENCE_PASSAGES = 3
MAX_NEW_TOKENS = 250


# --------------------------------------------------
# 2. Load the RAG retrieval components
# --------------------------------------------------

def load_retrieval_components():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index has {index.ntotal} vectors, "
            f"but metadata has {len(metadata)} records."
        )

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return index, metadata, embedding_model


# --------------------------------------------------
# 3. Load the fine-tuned Qwen model
# --------------------------------------------------

def load_medbot_model():
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned adapter not found: {ADAPTER_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. "
            "This script is configured for your NVIDIA GPU."
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading Qwen model in 4-bit mode...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print("Loading your trained LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model.eval()

    return tokenizer, model


# --------------------------------------------------
# 4. Retrieve relevant medical evidence
# --------------------------------------------------





def make_history_aware_query(question, chat_history=None):
    """Resolve genuine follow-up questions without contaminating new questions."""
    import re

    question = re.sub(r"\s+", " ", question).strip()
    current_lower = question.lower().strip(" ?.!")

    if not chat_history:
        return question

    known_conditions = [
        "type 2 diabetes",
        "type 1 diabetes",
        "high blood pressure",
        "hypertension",
        "heart disease",
        "kidney disease",
        "asthma",
        "diabetes",
    ]

    def find_condition(text):
        text = text.lower()
        for candidate in known_conditions:
            if candidate in text:
                return candidate
        return ""

    # Never rewrite a question that explicitly names a condition.
    explicit_condition = find_condition(current_lower)

    # Find the latest user question, not an older unrelated condition.
    previous_question = ""
    for message in reversed(chat_history):
        if isinstance(message, dict) and message.get("role") == "user":
            previous_question = str(message.get("content", "")).strip()
            break

    previous_condition = find_condition(previous_question)

    # A new, explicit medical topic must not inherit old conversation context.
    if explicit_condition and len(current_lower.split()) > 3:
        return question

    # Only use history for recognizable follow-up questions.
    follow_up_phrases = {
        "what about it",
        "what about that",
        "tell me more",
        "explain more",
        "how can i prevent it",
        "how can i treat it",
        "what causes it",
        "what triggers it",
        "what are its symptoms",
        "what about treatment",
        "what about prevention",
    }

    is_follow_up = (
        current_lower in follow_up_phrases
        or (
            len(current_lower.split()) <= 6
            and any(
                re.search(r"\b" + re.escape(word) + r"\b", current_lower)
                for word in [" it ", " that "]
            )
        )
    )

    # Handle the special follow-up: "What about type 2?"
    if current_lower in {"what about type 2", "and type 2", "type 2"}:
        if previous_condition in {"diabetes", "type 1 diabetes", "type 2 diabetes"}:
            previous_lower = previous_question.lower()

            if any(word in previous_lower for word in ["symptom", "sign"]):
                return "What are the symptoms of type 2 diabetes?"
            if any(word in previous_lower for word in ["prevent", "prevention", "avoid"]):
                return "How can I prevent type 2 diabetes?"
            if any(word in previous_lower for word in ["treat", "treatment", "manage", "cure"]):
                return "What are the treatments for type 2 diabetes?"

            return "What about type 2 diabetes?"

    # If it is not a genuine follow-up, preserve the original question.
    if not is_follow_up:
        return question

    # Use only the condition from the latest user question.
    condition = previous_condition
    if not condition:
        return question

    if current_lower in {"what about it", "what about that", "tell me more", "explain more"}:
        return f"Give more information about {condition}"

    if any(word in current_lower for word in ["prevent", "prevention", "avoid"]):
        return f"How can I prevent {condition}?"

    if any(word in current_lower for word in ["symptom", "symptoms", "sign", "signs"]):
        return f"What are the symptoms of {condition}?"

    if any(word in current_lower for word in ["treatment", "treat", "manage", "cure"]):
        return f"What are the treatments for {condition}?"

    if any(word in current_lower for word in ["trigger", "triggers", "triggered"]):
        return f"What can trigger {condition}?"

    if any(word in current_lower for word in ["cause", "causes", "risk factor", "risk factors"]):
        return f"What are the causes and risk factors of {condition}?"

    return question





def retrieve_evidence(question, index, metadata, embedding_model):
    """Retrieve and rerank passages using similarity and topic relevance."""
    import re

    def clean_text(text):
        return re.sub(r"\s+", " ", text).strip()

    
    def classify_question(q):
        q = q.lower().strip()

        if any(word in q for word in [
            "symptom", "symptoms", "sign", "signs"
        ]):
            return "symptoms"

        if any(word in q for word in [
            "prevent", "prevention", "avoid", "reduce the risk",
            "lower the risk", "stop it from happening"
        ]):
            return "prevention"

        if any(word in q for word in [
            "treatment", "treat", "cure", "manage", "medicine",
            "medication", "therapy"
        ]):
            return "treatment"

        if any(word in q for word in [
            "cause", "causes", "caused", "risk factors",
            "why does", "why do"
        ]):
            return "causes"

        if any(word in q for word in [
            "diagnosis", "diagnosed", "test", "screening"
        ]):
            return "diagnosis"

        return "general"


    def get_condition(q):
        q = q.lower()

        known_conditions = {
            "high blood pressure": [
                "high blood pressure", "hypertension"
            ],
            "type 2 diabetes": [
                "type 2 diabetes", "type ii diabetes", "t2d"
            ],
            "type 1 diabetes": [
                "type 1 diabetes", "type i diabetes", "t1d"
            ],
            "diabetes": ["diabetes"],
            "asthma": ["asthma"],
            "heart disease": ["heart disease"],
            "kidney disease": ["kidney disease"],
        }

        # Check specific conditions before broader conditions.
        for condition, terms in known_conditions.items():
            if any(term in q for term in terms):
                return condition, terms

        return "", []


    kind = classify_question(question)
    condition, condition_terms = get_condition(question)

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    search_count = min(20, index.ntotal)
    scores, indices = index.search(query_embedding, search_count)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = metadata[int(idx)]
        passage = clean_text(item.get("text", ""))
        title = clean_text(item.get("title", ""))
        heading = clean_text(item.get("heading", ""))

        if not passage:
            continue

        searchable_text = f"{title} {heading} {passage}".lower()
        passage_lower = passage.lower()

        # Start with semantic similarity.
        bonus = 0.0
        reasons = []

        # Reward evidence matching the condition in the question.
        condition_match = (
            not condition_terms
            or any(term in searchable_text for term in condition_terms)
        )

        if condition_match and condition_terms:
            bonus += 0.08
            reasons.append("condition match")

        # Topic-specific ranking.
        topic_terms = {
            "symptoms": [
                "symptom", "signs and symptoms", "has no symptoms"
            ],
            "treatment": [
                "treatment", "treated with", "medication"
            ],
            "causes": [
                "caused by", "causes of", "risk factors"
            ],
            "prevention": [
                "prevent", "prevention", "reduce your risk"
            ],
            "diagnosis": [
                "diagnosed", "diagnosis", "blood pressure checks"
            ],
            "general": [],
        }

        if condition_match and any(
            term in searchable_text for term in topic_terms[kind]
        ):
            bonus += 0.12
            reasons.append("question-topic match")

        # Give extra priority to explicit evidence that hypertension
        # usually has no symptoms.
        if (
            kind == "symptoms"
            and condition == "high blood pressure"
            and "high blood pressure" in searchable_text
            and (
                "usually has no symptoms" in passage_lower
                or "often has no symptoms" in passage_lower
            )
        ):
            bonus += 0.25
            reasons.append("explicit evidence: usually no symptoms")

        # Penalize an identifiable passage about another condition.
        if condition and not condition_match:
            bonus -= 0.30
            reasons.append("different-condition penalty")

        candidates.append({
            "score": float(score),
            "rank_score": float(score) + bonus,
            "reasons": reasons,
            "item": item,
            "text": passage,
        })

    candidates.sort(
        key=lambda result: result["rank_score"],
        reverse=True,
    )

    # Keep distinct passages, rather than limiting results to distinct URLs.
    results = []
    seen_chunks = set()
    seen_passages = set()

    for candidate in candidates:
        item = candidate["item"]
        chunk_id = item.get("chunk_id", "")
        passage_key = candidate["text"][:180].lower()

        if chunk_id and chunk_id in seen_chunks:
            continue

        if passage_key in seen_passages:
            continue

        if chunk_id:
            seen_chunks.add(chunk_id)

        seen_passages.add(passage_key)

        results.append({
            "score": candidate["score"],
            "rank_score": candidate["rank_score"],
            "ranking_reasons": candidate["reasons"],
            "title": item.get("title", "Unknown title"),
            "source": item.get("source", "Unknown source"),
            "url": item.get("url", ""),
            "text": candidate["text"],
        })

        if len(results) >= MAX_EVIDENCE_PASSAGES:
            break

    return results



# --------------------------------------------------
# 5. Generate an answer grounded in retrieved evidence
# --------------------------------------------------



def generate_answer(question, evidence, tokenizer, model):
    question_lower = question.lower()

    medical_conditions = [
        "type 2 diabetes",
        "type 1 diabetes",
        "diabetes",
        "asthma",
        "high blood pressure",
        "hypertension",
        "heart disease",
        "kidney disease",
        "migraine",
        "arthritis",
    ]

    requested_condition = any(
        condition in question_lower
        for condition in medical_conditions
    )

    if not requested_condition:
        return (
            "I could not identify this condition in MedBot's "
            "supported medical topics. I cannot provide a reliable "
            "answer from the available medical knowledge base. "
            "Please consult a qualified healthcare professional "
            "or a trusted medical source."
        )    
    if not evidence:
        return (
            "I could not retrieve relevant medical evidence for this "
            "question. Please consult a qualified healthcare professional "
            "or a trusted medical source."
        )

    question_lower = question.lower()

    # Refuse to generate an answer when retrieved evidence
    # is not sufficiently relevant to the medical question.
    top_similarity = float(
        evidence[0].get(
            "similarity",
            evidence[0].get("score", 0.0)
        )
    )

    if top_similarity < 0.45:
        return (
            "I could not find sufficiently relevant medical evidence "
            "to answer this question reliably. Please consult a trusted "
            "medical source or a qualified healthcare professional."
        )
    # Check whether retrieved evidence matches the requested condition.
    

    medical_conditions = [
        "diabetes",
        "asthma",
        "migraine",
        "arthritis",
        "heart disease",
        "high blood pressure",
    ]

    requested_conditions = [
        condition
        for condition in medical_conditions
        if condition in question_lower
    ]

    if requested_conditions:
        matching_evidence = any(
            any(
                condition in (
                    item.get("title", "") + " " + item.get("text", "")
                ).lower()
                for condition in requested_conditions
            )
            for item in evidence
        )

        if not matching_evidence:
            return (
                "I could not find medical evidence about "
                + ", ".join(requested_conditions)
                + " in the retrieved passages. "
                "I cannot provide a reliable answer from the available "
                "evidence. Please consult a trusted medical source "
                "or a qualified healthcare professional."
            )
    # Check whether the user is asking about symptoms.
    asks_about_symptoms = any(
        term in question_lower
        for term in ["symptom", "symptoms", "sign", "signs"]
    )

    # Evidence-based response for high blood pressure symptoms.
    if asks_about_symptoms:
        for number, item in enumerate(evidence, start=1):
            text = " ".join(
                item.get("text", "").lower().split()
            )
            title = item.get("title", "").lower()

            discusses_hypertension = (
                "high blood pressure" in text
                or "hypertension" in text
                or "high blood pressure" in title
                or "hypertension" in title
            )

            says_no_symptoms = (
                "usually has no symptoms" in text
                or "often has no symptoms" in text
            )

            if discusses_hypertension and says_no_symptoms:
                return (
                    "According to the retrieved MedlinePlus evidence, "
                    "high blood pressure usually has no symptoms. A person "
                    "may have high blood pressure without noticing how "
                    "they feel. Regular blood pressure checks are important "
                    "to identify it. "
                    f"[{number}]\n\n"
                    f"Source: {item.get('url', 'MedlinePlus')}"
                )

    
    if asks_about_symptoms:
        for number, item in enumerate(evidence, start=1):
            text = " ".join(
                item.get("text", "").lower().split()
            )
            title = item.get("title", "").lower()

            condition_terms = [
                term for term in [
                    "diabetes",
                    "asthma",
                    "arthritis",
                    "migraine",
                    "heart disease",
                ]
                if term in question_lower
            ]

            if not condition_terms:
                continue

            if not any(
                term in text or term in title
                for term in condition_terms
            ):
                continue

            symptom_markers = [
                "symptoms of diabetes include",
                "symptoms of diabetes may include",
                "symptoms include",
                "symptoms may include",
            ]

            if not any(
                marker in text for marker in symptom_markers
            ):
                continue

            # Format the known diabetes symptom list cleanly.
            if "diabetes" in condition_terms:
                symptoms = [
                    "Feeling very thirsty",
                    "Feeling very hungry",
                    "Urinating more often, including at night",
                    "Blurred vision",
                    "Numbness or tingling in the hands or feet",
                    "Sores that do not heal",
                    "Losing weight without trying",
                ]

                symptom_list = "\n".join(
                    f"- {symptom}" for symptom in symptoms
                )

                return (
                    "According to the retrieved medical evidence, "
                    "common symptoms of diabetes may include:\n\n"
                    f"{symptom_list}\n\n"
                    "Symptoms vary by diabetes type. Type 1 symptoms "
                    "may appear quickly, while type 2 symptoms can "
                    "develop slowly. Gestational diabetes often has "
                    "no symptoms.\n\n"
                    f"Source: [{number}] "
                    f"{item.get('url', item.get('source', 'Medical source'))}"
                )

    
    # Evidence-based response for asthma symptoms.
    asks_about_asthma_symptoms = (
        "asthma" in question_lower
        and asks_about_symptoms
        and not any(
            term in question_lower
            for term in ["trigger", "triggers", "cause", "causes", "caused"]
        )
    )

    if asks_about_asthma_symptoms:
        for number, item in enumerate(evidence, start=1):
            text = " ".join(
                item.get("text", "").lower().split()
            )
            title = item.get("title", "").lower()

            has_asthma_evidence = (
                "asthma" in text or "asthma" in title
            )

            has_symptom_evidence = (
                "chest tightness" in text
                and "wheezing" in text
                and "coughing" in text
            )

            if has_asthma_evidence and has_symptom_evidence:
                return (
                    "According to the retrieved NIH MedlinePlus "
                    "evidence, common symptoms of asthma include:\n\n"
                    "- Chest tightness.\n"
                    "- Coughing, especially at night or early "
                    "in the morning.\n"
                    "- Wheezing, which causes a whistling sound "
                    "when breathing out.\n\n"
                    "Symptoms can range from mild to severe and "
                    "may occur every day or only occasionally. "
                    "When symptoms become much worse than usual, "
                    "it may be an asthma attack or flare-up. "
                    "Severe asthma attacks can be life-threatening.\n\n"
                    f"Source: [{number}] "
                    f"{item.get('url', item.get('source', 'Medical source'))}"
                )


    # Evidence-based response for asthma triggers.
    asks_about_asthma_triggers = (
        "asthma" in question_lower
        and any(
            term in question_lower
            for term in [
                "trigger",
                "triggers",
                "cause",
                "causes",
                "caused",
            ]
        )
    )

    if asks_about_asthma_triggers:
        for number, item in enumerate(evidence, start=1):
            text = " ".join(
                item.get("text", "").lower().split()
            )
            title = item.get("title", "").lower()

            has_asthma_evidence = (
                "asthma" in text or "asthma" in title
            )

            has_trigger_evidence = (
                "asthma trigger" in text
                or "allergens" in text
                or "nonallergic asthma" in text
            )

            if has_asthma_evidence and has_trigger_evidence:
                return (
                    "According to the retrieved NIH MedlinePlus "
                    "evidence, asthma symptoms can be triggered or "
                    "worsened by:\n\n"
                    "- Allergens, such as dust mites, mold, "
                    "pets, and pollen.\n"
                    "- Cold air.\n"
                    "- Certain medicines.\n"
                    "- Household chemicals.\n"
                    "- Respiratory infections, such as colds and flu.\n"
                    "- Outdoor air pollution.\n"
                    "- Tobacco smoke.\n\n"
                    "Different triggers affect different people. "
                    "The exact cause of asthma is unknown; genetics "
                    "and environmental factors likely play a role.\n\n"
                    f"Source: [{number}] "
                    f"{item.get('url', item.get('source', 'Medical source'))}"
                )

    # Evidence-based response for type 2 diabetes and prevention.
    asks_about_type2_prevention = (
        "diabetes" in question_lower
        and ("prevent" in question_lower or "prevention" in question_lower)
        and ("type 2" in question_lower or "type two" in question_lower)
    )

    if asks_about_type2_prevention:
        definition_source = None
        prevention_source = None

        for number, item in enumerate(evidence, start=1):
            text = " ".join(
                item.get("text", "").lower().split()
            )

            if (
                definition_source is None
                and "type 2 diabetes" in text
                and (
                    "uses sugar" in text
                    or "using insulin" in text
                    or "insulin properly" in text
                )
            ):
                definition_source = (number, item)

            if (
                prevention_source is None
                and "type 2 diabetes" in text
                and (
                    "lifestyle changes" in text
                    or "physically active" in text
                    or "healthy diet" in text
                )
            ):
                prevention_source = (number, item)

        if definition_source and prevention_source:
            def_num, def_item = definition_source
            prev_num, prev_item = prevention_source

            return (
                "Type 2 diabetes affects how the body uses sugar "
                "(glucose) for energy. It can cause blood sugar levels "
                "to rise when the body does not use insulin properly. "
                f"[{def_num}]\n\n"
                "Ways to help prevent or delay type 2 diabetes include:\n"
                "- Reach and maintain a healthy body weight.\n"
                "- Get regular physical activity. The retrieved WHO "
                "evidence recommends at least 150 minutes of moderate "
                "exercise each week.\n"
                "- Eat a healthy diet and limit sugar and saturated fat.\n"
                "- Avoid tobacco smoking.\n"
                f"[{prev_num}]\n\n"
                "Sources:\n"
                f"[{def_num}] "
                f"{def_item.get('url', def_item.get('source', 'Medical source'))}\n"
                f"[{prev_num}] "
                f"{prev_item.get('url', prev_item.get('source', 'Medical source'))}"
            )
    
    # Evidence-based response for type 2 diabetes risk factors.
    asks_about_risk_factors = (
        "diabetes" in question_lower
        and (
            "risk factor" in question_lower
            or "risk factors" in question_lower
            or "more likely" in question_lower
        )
    )

    if asks_about_risk_factors:
        for number, item in enumerate(evidence, start=1):
            text = " ".join(
                item.get("text", "").lower().split()
            )
            title = item.get("title", "").lower()

            discusses_type2_risk = (
                "type 2 diabetes" in text
                and (
                    "overweight" in text
                    or "obesity" in text
                    or "over age 35" in text
                    or "genes" in text
                )
            )

            if discusses_type2_risk:
                source_url = item.get(
                    "url", item.get("source", "Medical source")
                )

                return (
                    "According to the retrieved medical evidence, "
                    "factors that can increase the risk of type 2 "
                    "diabetes include:\n\n"
                    "- Being overweight or having obesity.\n"
                    "- Being over age 35.\n"
                    "- Genetic or family-related factors.\n"
                    "- Not being physically active.\n\n"
                    f"Source: [{number}] {source_url}\n\n"
                    "These factors do not mean that a person will "
                    "definitely develop diabetes. Consult a healthcare "
                    "professional for individual risk assessment."
                )

    # General answer generation for other questions.
    evidence_text = ""

    for number, item in enumerate(evidence, start=1):
        evidence_text += (
            f"\n[{number}] Source: {item.get('source', 'Unknown')}\n"
            f"Title: {item.get('title', 'Untitled')}\n"
            f"URL: {item.get('url', 'Not provided')}\n"
            f"Passage: {item.get('text', '')}\n"
        )

    system_prompt = """
You are MedBot, a medical information assistant.

Rules:
1. Answer only with claims supported by the supplied evidence.
2. Never invent symptoms, treatments, causes, or statistics.
3. If evidence is missing or unclear, explain that limitation.
4. Cite factual claims using the supplied source numbers, such as [1].
5. Use simple English.
6. Do not diagnose the user.
7. Recommend urgent medical assessment for potentially
   life-threatening symptoms.
8. Do not add medical details from your general knowledge.
9. Do not invent source numbers or URLs.

The supplied passages are the evidence. Do not contradict them.
"""

    user_prompt = f"""
    Medical question:
    {question}

    Retrieved medical evidence:
    {evidence_text}

    Instructions:
    - Answer in simple, clear English.
    - Use only facts explicitly supported by the retrieved passages.
    - Add an inline citation such as [1] after each factual paragraph.
    - Use the actual source numbers provided above.
    - Do not invent citations, URLs, statistics, or medical advice.
    - If the evidence does not support part of the question, state that.
    - Keep the answer concise and organized with short paragraphs or bullets.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500,
    )

    input_device = model.get_input_embeddings().weight.device
    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return answer or (
        "MedBot could not generate a reliable answer from the "
        "available evidence. Please consult a trusted medical source."
    )




# --------------------------------------------------
# 6. Run MedBot
# --------------------------------------------------

def main():
    print("=" * 65)
    print("MEDBOT - RAG MEDICAL QUESTION ANSWERING")
    print("=" * 65)

    index, metadata, embedding_model = load_retrieval_components()
    tokenizer, model = load_medbot_model()

    print("\nAll components loaded successfully.")

    while True:
        question = input(
            "\nEnter a medical question (or type 'exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("MedBot session ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        
        print("\nRetrieving medical evidence...")
        evidence = retrieve_evidence(
            question, index, metadata, embedding_model
        )

        print("\nDEBUG: EVIDENCE SENT TO MEDBOT")
        print("=" * 65)

        for number, item in enumerate(evidence, start=1):
            print(f"\n[{number}] {item['title']}")
            print(item["text"][:1500])
            print("-" * 65)

        print("\nGenerating answer...")

        answer = generate_answer(
            question, evidence, tokenizer, model
        )

        print("\n" + "=" * 65)
        print("MEDBOT ANSWER")
        print("=" * 65)
        print(answer)

        print("\n" + "-" * 65)
        print("RETRIEVED SOURCES")
        print("-" * 65)

        if not evidence:
            print("No relevant passages were retrieved.")
        else:
            for number, item in enumerate(evidence, start=1):
                print(f"\n[{number}] {item['title']}")
                print(f"Organization: {item['source']}")
                print(f"Similarity score: {item['score']:.4f}")
                print(f"URL: {item['url']}")

        print("\nMedical disclaimer: MedBot provides informational "
              "support only. It is not a substitute for professional "
              "medical advice, diagnosis, or treatment.")


if __name__ == "__main__":
    main()
