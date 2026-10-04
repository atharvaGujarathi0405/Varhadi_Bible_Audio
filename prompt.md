You are the lead engineer for my final-year project:

PROJECT:
KisaanDost — A Region-Adaptive Multimodal AI Assistant for Varhadi-Speaking Farmers in Vidarbha, Maharashtra.

IMPORTANT:
This is an existing project repository. DO NOT rebuild the project from scratch and DO NOT delete, replace, or unnecessarily refactor existing working code.

First inspect the entire repository and understand what already exists.

==================================================
1. PROJECT GOAL
==================================================

Build a research-oriented, production-quality prototype of KisaanDost for Varhadi-speaking farmers in Vidarbha, Maharashtra.

The system should eventually support:

1. Varhadi/Marathi voice input
2. Speech-to-text using an Indic ASR model
3. Varhadi linguistic normalization/adaptation
4. Agricultural intent and entity extraction
5. Vidarbha-specific agricultural RAG
6. Crop disease/pest image analysis
7. Grounded agricultural response generation
8. Marathi/Varhadi-compatible TTS
9. React web/mobile-friendly interface
10. Planned IVR integration
11. Authentication and role-based access
12. AI safety and security controls
13. Evaluation and experiment tracking

PRIMARY RESEARCH CONTRIBUTION:

Adapt/evaluate a pretrained Marathi ASR model for low-resource Varhadi speech and compare:

    Pretrained Marathi ASR
              VS
    Varhadi-adapted ASR

using the same unseen test dataset and metrics such as WER and CER.

Do NOT claim that we are creating a new language or training an ASR model from scratch.

==================================================
2. EXISTING WORK — PRESERVE IT
==================================================

The repository already contains a Varhadi audio downloader and ASR baseline work.

Existing downloader:

- downloader.py
- browser.py
- chapters.py
- config.py
- progress.py
- requirements.txt
- progress.json
- failed.json
- chapter_metadata.json
- tests/
- test_one_chapter.py
- audio/

The downloader collects VAHNT (Varhadi-Nagpuri New Testament) audio.

Current dataset status:

- 89 / 260 chapters downloaded
- approximately 162 MB
- approximately 706 minutes
- MP3 audio
- no verified transcripts
- no speaker IDs
- no timestamps

IMPORTANT:
VAHNT Bible audio is GENERAL VARHADI SPEECH.
It is NOT agricultural speech.

It can be used for:
- raw ASR baseline
- feasibility testing
- linguistic/error analysis

It must NOT automatically be treated as the final agricultural training dataset.

Existing ASR baseline:

asr_baseline/
    config.py
    audio_utils.py
    model.py
    transcribe.py
    evaluate.py
    error_analysis.py
    utils.py

scripts/
    validate_asr_dataset.py
    run_asr_baseline.py
    generate_asr_report.py

asr_metadata.csv

asr_outputs/

requirements-asr.txt

Current model:

ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large

This is the current pretrained Marathi ASR baseline.

Current environment:

- WSL2
- Python 3.12.3
- AI4Bharat NeMo fork
- NeMo 1.23.0rc0
- PyTorch CPU
- no NVIDIA GPU

Existing fixes include:
- HuggingFace gated-model authentication
- direct checkpoint download using hf_hub_download
- ASRModel.restore_from()
- NumPy compatibility shim
- decoder output unwrapping
- separate ASR environment
- independent predictions.csv resume mechanism

Current real result:

A 30-second Varhadi audio sample was successfully transcribed.

Full ~8-minute files currently cause CPU memory exhaustion.

DO NOT remove or rewrite these fixes.

The next immediate ASR engineering task is chunked inference.

==================================================
3. DEVELOPMENT RULES
==================================================

Before changing anything:

1. Inspect the repository.
2. Read README.md.
3. Read all existing ASR files.
4. Read downloader files.
5. Run the existing tests.
6. Identify the current architecture.
7. Create a short implementation plan.
8. Only then start modifying code.

NEVER:
- delete working code
- replace the downloader
- rebuild the ASR pipeline unnecessarily
- fabricate WER/CER
- fabricate transcripts
- claim agricultural training data exists when it does not
- claim Varhadi ASR adaptation is complete before training
- expose API keys
- hard-code secrets
- put model/API credentials in frontend code
- introduce unnecessary frameworks
- use Streamlit as the final frontend

The final frontend should be React.

Use:
- React
- TailwindCSS
- Material UI where useful
- FastAPI
- Python
- PostgreSQL
- vector database such as ChromaDB/Qdrant
- LangChain only where it provides real value
- modern REST APIs
- clean modular architecture

==================================================
4. TARGET ARCHITECTURE
==================================================

Implement this architecture:

                    FARMER
                       |
          +------------+------------+
          |                         |
      Smartphone                  IVR
          |                         |
          +------------+------------+
                       |
              Voice / Text / Image
                       |
                Input Validation
                       |
             +---------+---------+
             |         |         |
            ASR      Vision     Text
             |         |         |
             +---------+---------+
                       |
             Varhadi Adaptation
                       |
             Intent + Entity Layer
                       |
          +------------+-------------+
          |            |             |
         RAG         Vision       Services
          |            |             |
          +------------+-------------+
                       |
                      LLM
                       |
                 Safety Check
                       |
              Response Generation
                       |
                 Varhadi/Marathi
                       |
                      TTS
                       |
                     FARMER

Security should be a cross-cutting layer around the entire system.

==================================================
5. PHASE 0 — REPOSITORY AUDIT
==================================================

Start here.

Inspect:

- directory structure
- current Python environment
- package files
- frontend if any
- backend if any
- downloader
- ASR baseline
- tests
- configuration
- environment variables
- datasets
- documentation

Then report:

A. What already works
B. What is partially implemented
C. What is missing
D. What must not be changed
E. Recommended implementation order

Do not modify code during the audit unless absolutely necessary.

==================================================
6. PHASE 1 — ASR BASELINE
==================================================

Complete the current Experiment 0.

Goal:

Run the pretrained Marathi IndicConformer on Varhadi audio without:

- LLM correction
- dictionary correction
- translation
- Varhadi normalization
- fine-tuning

The raw output must remain raw.

Implement robust chunked inference.

Requirements:

- configurable chunk duration
- configurable overlap
- safe memory usage
- WAV conversion
- 16 kHz mono
- chunk metadata
- ordered reconstruction
- independent resume
- failure recovery
- logging
- no modification to downloader progress.json

Example:

audio file
    |
    +-- chunk 001
    +-- chunk 002
    +-- chunk 003
    ...
    |
    +-- ASR
    |
    +-- ordered transcript

Make chunking configurable through config/environment/CLI.

Do not blindly concatenate overlapping text.
Implement a documented stitching strategy.

Test chunking using short fixtures before running all 89 chapters.

==================================================
7. PHASE 2 — BASELINE DATASET PIPELINE
==================================================

Create a clean dataset structure.

Example:

data/
    raw/
    processed/
    transcripts/
    metadata/
    splits/
    evaluation/

Maintain metadata:

audio_path
speaker_id
district
duration
language
dialect
domain
transcript
consent_status
source
license
split

IMPORTANT:

For current Bible audio:

speaker_id = unknown
reference_text = unavailable

Do not fabricate references.

==================================================
8. PHASE 3 — VARHADI SPEECH DATASET
==================================================

Create tooling for collecting and validating native Varhadi recordings.

The target dataset should eventually contain:

audio + exact transcript

Prefer recordings covering:

- general conversation
- agricultural questions
- crop names
- pest names
- disease names
- irrigation
- fertilizer
- farming practices
- government agricultural services

Capture metadata:

speaker_id
district
age_group if consented
gender if consented
domain
duration
recording_quality
transcript
consent
source

Use pseudonymous speaker IDs.

Example:

VH_S001
VH_S002
VH_S003

Never use phone number/name as the model speaker ID.

Create train/validation/test split by SPEAKER, not random audio chunks.

Avoid speaker leakage.

==================================================
9. PHASE 4 — TRANSCRIPTION / GOLD DATA
==================================================

Build a workflow for manually verified transcripts.

Requirements:

- original audio preserved
- raw transcript
- corrected transcript
- reviewer
- review status
- timestamp if segmented
- provenance

Do not generate fake ground truth.

Only calculate WER/CER when reference_text is verified.

Support:

WER
CER
substitution
deletion
insertion

Use jiwer or another established implementation.

==================================================
10. PHASE 5 — VARHADI ASR ADAPTATION
==================================================

After sufficient paired data exists:

Start from:

ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large

Do transfer learning/fine-tuning rather than training from scratch.

Create:

experiments/
    baseline/
    varhadi_adaptation/
    evaluation/

Track:

- model version
- dataset version
- training configuration
- hyperparameters
- epoch
- learning rate
- batch size
- gradient accumulation
- checkpoint
- validation loss
- WER
- CER

Do not invent hyperparameters.
Inspect the actual AI4Bharat/NeMo model configuration and use compatible settings.

Because this machine has no GPU, design training so it can later be moved to a GPU environment if necessary.

Do not pretend CPU training is practical for a large model without measuring it.

==================================================
11. PRIMARY EXPERIMENT
==================================================

Implement a reproducible experiment:

EXPERIMENT A:
Pretrained Marathi ASR

EXPERIMENT B:
Varhadi-adapted ASR

Evaluate both on the EXACT SAME unseen test set.

Produce:

comparison.csv
comparison.json
comparison.md

Metrics:

- WER
- CER
- substitutions
- deletions
- insertions

Also analyze:

- Varhadi vocabulary errors
- agricultural terminology errors
- pronunciation-related errors
- code-switching errors
- proper nouns
- numbers
- crop/pest/disease terms

Never call the adapted model "better" automatically.
Report the actual measured results.

==================================================
12. PHASE 6 — VARHADI LINGUISTIC ADAPTATION
==================================================

Implement a separate post-ASR normalization layer.

This is NOT the ASR model.

Architecture:

raw ASR output
    |
text cleaning
    |
Varhadi vocabulary
    |
spelling / ASR variants
    |
agricultural terminology
    |
canonical representation
    |
intent + entities

Example:

{
  "intent": "crop_problem",
  "entities": {
      "crop": "cotton",
      "issue": "pest"
  },
  "location": "Vidarbha"
}

Do not use an LLM for every simple normalization operation.

Prefer:

1. deterministic rules
2. lexicon
3. terminology mapping
4. confidence handling
5. LLM fallback only where appropriate

Maintain a versioned Varhadi lexicon.

Each entry should have:

varhadi_term
canonical_term
marathi_term
english_term
category
source
confidence
notes

==================================================
13. PHASE 7 — AGRICULTURAL RAG
==================================================

Build a verified Vidarbha-focused agricultural knowledge base.

Start narrow.

Do NOT attempt all agriculture.

Initially focus on selected crops such as:

- cotton
- soybean
- orange
- tur

Confirm actual final crop scope before implementation.

Knowledge areas:

- crop practices
- diseases
- pests
- irrigation
- fertilizer
- government schemes
- basic agricultural guidance

Use trusted sources.

Prefer:

- ICAR
- agricultural universities
- government agriculture departments
- official government documents
- peer-reviewed agricultural sources

Do not automatically ingest random websites.

Every document must maintain provenance:

document_id
source
publication_date
last_updated
region
crop
topic
license
verification_status
review_date

Pipeline:

documents
    |
cleaning
    |
chunking
    |
embeddings
    |
vector database
    |
retrieval
    |
reranking if necessary
    |
LLM
    |
grounded answer

The LLM must not invent agricultural facts when relevant evidence is unavailable.

If retrieval confidence is low:

ask clarification
OR
say insufficient verified information is available
OR
escalate to expert

==================================================
14. PHASE 8 — AGRICULTURAL INTENT + ENTITY EXTRACTION
==================================================

Implement structured agricultural query understanding.

Possible intents:

crop_problem
disease_identification
pest_problem
fertilizer_query
irrigation_query
weather_query
crop_practice
government_scheme
market_information
general_agriculture
unknown

Entities:

crop
disease
pest
location
growth_stage
symptom
fertilizer
season
scheme

Start with deterministic/rule/LLM structured extraction where appropriate.

Keep this module independent from RAG.

==================================================
15. PHASE 9 — CROP IMAGE MODULE
==================================================

Build a modular vision service.

Input:

crop image

Pipeline:

image upload
    |
validation
    |
quality check
    |
vision model
    |
prediction + confidence
    |
RAG-supported guidance

Do NOT autonomously prescribe dangerous pesticide use.

For low confidence:

- request another image
- ask additional questions
- recommend expert verification

Do not claim a disease is definitively diagnosed unless the system is specifically validated for that purpose.

Keep the vision model replaceable.

Do not train a new vision model from scratch unless there is enough dataset and a clear research justification.

==================================================
16. PHASE 10 — LLM ORCHESTRATION
==================================================

Build a central orchestration service.

Input:

{
  text,
  language,
  dialect,
  intent,
  entities,
  location,
  retrieved_context,
  vision_result
}

LLM should generate:

- concise
- farmer-friendly
- grounded
- actionable but safe
- region-aware

Responses should not contain unsupported claims.

If retrieved context is insufficient, do not hallucinate.

==================================================
17. PHASE 11 — SAFETY LAYER
==================================================

Implement explicit AI safety controls.

ASR:

low confidence
    ->
ask user to repeat/confirm

Vision:

low confidence
    ->
request better image / expert verification

RAG:

no evidence
    ->
do not fabricate

Agricultural recommendation:

potentially harmful
    ->
verified source + safety information + escalation where appropriate

Ambiguous query:

    ->
clarification

Critical/uncertain case:

    ->
human/agricultural expert escalation

Add prompt-injection protection.

Treat user input and retrieved documents as untrusted data.

Separate system instructions from retrieved content.

==================================================
18. PHASE 12 — SECURITY
==================================================

Implement:

- authentication
- authorization
- RBAC
- secure sessions
- HTTPS-ready configuration
- CORS restrictions
- rate limiting
- request validation
- file-size limits
- MIME/type validation
- API-key protection
- environment variables
- no frontend secrets
- SQL injection prevention
- secure database access
- audit logging
- backup/recovery
- security headers where applicable

Roles:

FARMER
EXPERT
ADMIN

Farmer:
- ask questions
- upload images
- view own history

Expert:
- review flagged cases
- verify uncertain cases

Admin:
- manage knowledge base
- manage system configuration
- review security/audit logs

==================================================
19. PHASE 13 — PRIVACY / ETHICS
==================================================

Implement privacy by design.

For voice recordings:

- informed consent
- pseudonymous speaker IDs
- minimum data collection
- defined retention policy
- deletion mechanism where feasible
- source/license tracking

For farmer data:

- do not collect unnecessary PII
- protect phone numbers
- protect location
- restrict access
- encrypt sensitive data
- avoid unnecessary logging

Maintain dataset provenance.

==================================================
20. PHASE 14 — BACKEND
==================================================

Create a clean FastAPI backend.

Suggested structure:

backend/
    app/
        main.py
        config.py

        api/
            auth.py
            chat.py
            speech.py
            vision.py
            rag.py
            users.py
            admin.py

        services/
            asr_service.py
            normalization_service.py
            intent_service.py
            rag_service.py
            vision_service.py
            llm_service.py
            tts_service.py
            safety_service.py

        models/
        schemas/
        security/
        db/
        utils/

Do not force this exact structure if an existing backend already exists.
Adapt to the repository.

Create proper request/response schemas.

==================================================
21. PHASE 15 — FRONTEND
==================================================

Build a modern React frontend.

Do NOT use Streamlit as the final frontend.

Preferred:

React
TailwindCSS
Material UI where useful

Main screens:

1. Landing page
2. Farmer dashboard
3. Voice assistant
4. Text chat
5. Crop image upload
6. Conversation history
7. Agricultural result/source view
8. Profile/settings
9. Expert review dashboard
10. Admin knowledge-base dashboard

Design requirements:

- mobile-first
- simple
- large controls
- minimal text
- clear microphone button
- clear image upload
- regional-language friendly
- accessible to low digital-literacy users

Do not over-design.
Prioritize usability.

==================================================
22. PHASE 16 — CHAT FLOW
==================================================

Example:

Farmer speaks:

"Maza kapusala kide laglet"

Flow:

audio
 ->
ASR
 ->
raw transcript
 ->
Varhadi normalization
 ->
intent = crop_problem
 ->
crop = cotton
 ->
issue = pest
 ->
RAG retrieval
 ->
LLM
 ->
safety check
 ->
response
 ->
TTS

Expose intermediate processing only in developer/debug mode.

Farmer UI should remain simple.

==================================================
23. PHASE 17 — TTS
==================================================

Implement TTS as a replaceable service.

Initially use an available Marathi/Indic TTS model/service.

Do NOT claim it is Varhadi TTS until it has been adapted and evaluated.

Future:

Marathi/Indic TTS
+
Varhadi paired text/audio
->
Varhadi-adapted TTS

Track:

- intelligibility
- pronunciation
- naturalness
- native-speaker evaluation

==================================================
24. PHASE 18 — IVR
==================================================

Design the backend so IVR can be added later.

Do not tightly couple the core AI logic to the web frontend.

Architecture:

IVR provider
    |
webhook / websocket
    |
FastAPI
    |
ASR
    |
KisaanDost orchestration
    |
TTS
    |
IVR response

Initially implement an IVR-compatible API/interface or mock adapter if real telephony credentials are unavailable.

Do not spend excessive time on telephony before the core AI pipeline works.

==================================================
25. DATABASE
==================================================

Use PostgreSQL for application data.

Potential tables:

users
farmer_profiles
conversations
messages
voice_records
image_records
agricultural_queries
knowledge_documents
knowledge_chunks
expert_reviews
audit_logs
model_versions

Use migrations.

Do not store huge audio/image files directly in PostgreSQL unless there is a clear reason.

Use object storage/local storage abstraction.

==================================================
26. EVALUATION DASHBOARD
==================================================

Create an experiment/evaluation module.

Display:

ASR:
- WER
- CER
- substitutions
- deletions
- insertions

Intent:
- accuracy
- precision
- recall
- F1

Entities:
- precision
- recall
- F1

RAG:
- retrieval Recall@K
- groundedness
- answer correctness

Vision:
- accuracy
- precision
- recall
- F1
- confusion matrix

System:
- latency
- throughput
- task completion
- failure rate

Do not generate fake results.

Use "N/A" or "Pending" until real measurements exist.

==================================================
27. TESTING
==================================================

Maintain unit + integration tests.

At minimum test:

ASR:
- audio validation
- chunking
- transcription
- resume
- malformed audio

RAG:
- retrieval
- no-result handling
- source provenance

LLM:
- structured output
- unsupported question
- prompt injection

Vision:
- invalid image
- oversized image
- low-quality image
- low confidence

Security:
- unauthorized API
- invalid token
- rate limiting
- invalid file
- SQL injection attempt
- unauthorized KB modification

Backend:
- API validation
- database operations

Frontend:
- major user flows

Never remove existing passing tests.

==================================================
28. DOCUMENTATION
==================================================

Maintain:

README.md

docs/
    architecture.md
    setup.md
    dataset.md
    asr_baseline.md
    asr_adaptation.md
    rag.md
    security.md
    safety.md
    evaluation.md
    deployment.md
    api.md

Also maintain:

EXPERIMENT_LOG.md

For every experiment record:

date
objective
dataset
model
configuration
result
limitations
next step

==================================================
29. ENVIRONMENT / SECRETS
==================================================

Use:

.env

and .env.example

Never commit:

HF_TOKEN
API keys
database passwords
telephony credentials
LLM keys

Ensure .gitignore protects secrets.

==================================================
30. DEVELOPMENT PRIORITY
==================================================

Do NOT attempt to finish every module simultaneously.

Use this priority:

P0:
Existing ASR baseline
+
chunked inference
+
baseline evaluation pipeline

P1:
Varhadi dataset/transcription tooling
+
evaluation dataset

P2:
ASR fine-tuning
+
baseline vs adapted comparison

P3:
Varhadi normalization
+
agricultural intent/entity extraction

P4:
Vidarbha RAG

P5:
LLM orchestration

P6:
Vision

P7:
TTS

P8:
React frontend

P9:
Authentication/security

P10:
IVR prototype

P11:
Deployment

If time becomes limited, prioritize P0-P5 over cosmetic features.

==================================================
31. TWO-MONTH CONSTRAINT
==================================================

This is a final-year project with approximately two months remaining.

Therefore:

DO NOT build an enormous enterprise system.

Prioritize a strong research result over feature quantity.

The minimum defensible final project should be:

1. Varhadi dataset
2. Marathi ASR baseline
3. Varhadi ASR adaptation
4. WER/CER comparison
5. Varhadi normalization
6. Vidarbha agricultural RAG
7. Safe LLM response
8. Working React + FastAPI prototype
9. Security/privacy controls
10. Evaluation report

Vision/TTS/IVR should be modular and implemented according to remaining time.

==================================================
32. IMPORTANT RESEARCH INTEGRITY RULES
==================================================

Never fabricate:

- dataset size
- WER
- CER
- accuracy
- number of farmers
- user-study results
- model improvement
- training results
- expert validation
- agricultural effectiveness

If something has not been measured:

write:

PENDING

If something is unavailable:

write:

NOT AVAILABLE

If a result is qualitative:

label it:

QUALITATIVE OBSERVATION

Keep baseline and adapted results separate.

==================================================
33. FINAL DELIVERABLE
==================================================

The final repository should contain:

1. Working ASR baseline
2. Chunked inference
3. Dataset tooling
4. Gold-transcript/evaluation workflow
5. Varhadi ASR adaptation training pipeline
6. Baseline vs adapted evaluation
7. Varhadi normalization
8. Agricultural intent/entity extraction
9. Vidarbha RAG
10. Safety layer
11. Security layer
12. FastAPI backend
13. React frontend
14. Vision service
15. TTS service
16. IVR adapter/prototype
17. Database
18. Tests
19. Documentation
20. Experiment logs

==================================================
34. HOW YOU SHOULD WORK
==================================================

Work incrementally.

For every phase:

1. Inspect existing implementation.
2. Explain what you found.
3. Make the smallest required changes.
4. Run tests.
5. Verify functionality.
6. Update documentation.
7. Report:
   - files changed
   - what changed
   - tests run
   - results
   - known limitations
   - next recommended step

NEVER silently move to a huge batch operation.

Before expensive operations such as:

- 89-file ASR inference
- model fine-tuning
- downloading large models
- large dataset processing

first provide the command and explain expected resource requirements.

==================================================
START NOW
==================================================

Do NOT immediately start coding.

First:

1. Inspect the complete repository.
2. Identify the existing downloader and ASR implementation.
3. Run the existing tests.
4. Give me:
   - current architecture
   - completed components
   - incomplete components
   - files that must be preserved
   - risks/issues
   - exact P0/P1/P2 implementation plan

Then wait for my approval before making major changes.

The goal is not simply to make a demo.

The goal is to build a technically defensible final-year research project where every experimental claim can be reproduced and every reported metric comes from real data.