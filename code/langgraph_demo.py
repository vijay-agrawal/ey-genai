import streamlit as st
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

###############################################################################
# 1) AZURE OPENAI CLIENT SETUP
###############################################################################
@st.cache_resource
def get_azure_client():
    """Initialize Azure OpenAI client with environment variables."""
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

###############################################################################
# 2) DEFINE STATE
###############################################################################
class HospitalState(TypedDict):
    symptoms: str
    nurse_specialty: str
    vitals: str
    doctor_specialty: str
    prescription: str
    messages: Annotated[list, add]

###############################################################################
# 3) LLM HELPER FUNCTION
###############################################################################
def call_llm(client: AzureOpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call Azure OpenAI and return the response."""
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

###############################################################################
# 4) NODE FUNCTIONS WITH LLM REASONING
###############################################################################
def attendant_node(state: HospitalState) -> dict:
    """Attendant uses LLM to route patient to appropriate nurse."""
    client = get_azure_client()
    symptoms = state.get("symptoms", "")
    
    system_prompt = """You are a hospital attendant. Your job is to analyze patient symptoms and route them to the appropriate specialist nurse.
    
Available nurses:
- Heart Nurse (for cardiac, chest pain, heart-related issues)
- ENT Nurse (for ear, nose, throat issues)
- Ortho Nurse (for bone, joint, muscle issues)
- Pediatric Nurse (for children's health issues)

Respond with ONLY the nurse specialty name (e.g., "Heart Nurse", "ENT Nurse", "Ortho Nurse", or "Pediatric Nurse")."""
    
    user_prompt = f"Patient symptoms: {symptoms}\n\nWhich nurse should handle this patient?"
    
    nurse_specialty = call_llm(client, system_prompt, user_prompt)
    
    message = f"👔 Attendant: Based on symptoms '{symptoms}', routing to {nurse_specialty}"
    
    return {
        "nurse_specialty": nurse_specialty,
        "messages": [message]
    }

def nurse_node(state: HospitalState) -> dict:
    """Nurse uses LLM to measure and record relevant vitals."""
    client = get_azure_client()
    nurse_specialty = state.get("nurse_specialty", "")
    symptoms = state.get("symptoms", "")
    
    system_prompt = f"""You are a {nurse_specialty} in a hospital. Based on the patient's symptoms, determine what vitals and measurements you would take.

Provide realistic vital signs and measurements relevant to your specialty. Format your response as a simple list of measurements.
For example: "Blood Pressure: 130/80 mmHg, Pulse: 75 bpm, Temperature: 98.6°F" """
    
    user_prompt = f"Patient symptoms: {symptoms}\n\nWhat vitals would you measure and what would typical readings be?"
    
    vitals = call_llm(client, system_prompt, user_prompt)
    
    message = f"👨‍⚕️ Nurse ({nurse_specialty}): Measured vitals - {vitals}"
    
    return {
        "vitals": vitals,
        "messages": [message]
    }

def doctor_node(state: HospitalState) -> dict:
    """Doctor uses LLM to diagnose and prescribe treatment."""
    client = get_azure_client()
    nurse_specialty = state.get("nurse_specialty", "")
    symptoms = state.get("symptoms", "")
    vitals = state.get("vitals", "")
    
    # Determine doctor specialty based on nurse
    system_prompt_routing = f"""Based on the nurse specialty "{nurse_specialty}", what doctor specialty should see this patient?
    
Respond with ONLY the doctor specialty (e.g., "Cardiologist", "ENT Specialist", "Orthopedic Surgeon", "Pediatrician")."""
    
    doctor_specialty = call_llm(client, system_prompt_routing, f"Nurse specialty: {nurse_specialty}")
    
    # Doctor prescribes treatment
    system_prompt_prescription = f"""You are a {doctor_specialty}. Review the patient's symptoms and vitals, then provide a prescription or treatment plan.

Keep your response concise - just the medication name and dosage or treatment recommendation."""
    
    user_prompt = f"Patient symptoms: {symptoms}\nVitals: {vitals}\n\nWhat is your diagnosis and prescription?"
    
    prescription = call_llm(client, system_prompt_prescription, user_prompt)
    
    message = f"👨‍⚕️ Doctor ({doctor_specialty}): Diagnosis and prescription - {prescription}"
    
    return {
        "doctor_specialty": doctor_specialty,
        "prescription": prescription,
        "messages": [message]
    }

def pharmacist_node(state: HospitalState) -> dict:
    """Pharmacist uses LLM to provide medication instructions."""
    client = get_azure_client()
    prescription = state.get("prescription", "")
    
    system_prompt = """You are a hospital pharmacist. Review the doctor's prescription and provide clear instructions for the patient.

Include:
1. How to take the medication
2. Important warnings or side effects
3. When they can pick it up

Keep it concise and patient-friendly."""
    
    user_prompt = f"Doctor's prescription: {prescription}\n\nProvide patient instructions:"
    
    instructions = call_llm(client, system_prompt, user_prompt)
    
    message = f"💊 Pharmacist: {instructions}"
    
    return {
        "messages": [message]
    }

###############################################################################
# 5) BUILD THE WORKFLOW GRAPH
###############################################################################
def create_hospital_workflow():
    """Create and compile the hospital workflow graph."""
    workflow = StateGraph(HospitalState)
    
    # Add nodes
    workflow.add_node("attendant", attendant_node)
    workflow.add_node("nurse", nurse_node)
    workflow.add_node("doctor", doctor_node)
    workflow.add_node("pharmacist", pharmacist_node)
    
    # Set entry point
    workflow.set_entry_point("attendant")
    
    # Connect the nodes in sequence
    workflow.add_edge("attendant", "nurse")
    workflow.add_edge("nurse", "doctor")
    workflow.add_edge("doctor", "pharmacist")
    workflow.add_edge("pharmacist", END)
    
    # Compile the graph
    return workflow.compile()

###############################################################################
# 6) STREAMLIT APP
###############################################################################
st.set_page_config(page_title="Hospital Multi-Agent Demo", page_icon="🏥")

st.title("🏥 Hospital Multi-Agent Demo with LLM Reasoning")
st.write("AI-powered hospital workflow: **Attendant → Nurse → Doctor → Pharmacist**")

# Check environment variables
if not all([os.getenv("AZURE_OPENAI_ENDPOINT"), 
            os.getenv("AZURE_OPENAI_API_KEY"), 
            os.getenv("AZURE_OPENAI_API_VERSION")]):
    st.error("⚠️ Missing Azure OpenAI environment variables. Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_API_VERSION")
    st.stop()

st.markdown("---")

# Ask user for symptoms
user_symptoms = st.text_area(
    "Enter your symptoms here:", 
    placeholder="e.g., I have chest pain and difficulty breathing",
    height=100
)

col1, col2 = st.columns([1, 3])
with col1:
    run_button = st.button("🔍 Get Prescription", type="primary")

if run_button:
    if not user_symptoms.strip():
        st.warning("⚠️ Please enter some symptoms.")
    else:
        with st.spinner("🤖 AI agents processing through hospital workflow..."):
            try:
                # Create the workflow
                app = create_hospital_workflow()
                
                # Run the multi-agent workflow with the user's symptoms
                initial_state = {
                    "symptoms": user_symptoms,
                    "nurse_specialty": "",
                    "vitals": "",
                    "doctor_specialty": "",
                    "prescription": "",
                    "messages": []
                }
                
                result = app.invoke(initial_state)
                
                st.success("✅ Workflow completed successfully!")
                
                st.markdown("### 📋 AI Agent Workflow Steps:")
                for msg in result.get("messages", []):
                    st.info(msg)
                
                st.markdown("### 💊 Final Prescription:")
                st.success(f"**{result.get('prescription', 'No prescription generated')}**")
                
                # Additional details in expander
                with st.expander("📊 View Complete AI-Powered Patient Journey"):
                    st.write(f"**Initial Symptoms:** {result.get('symptoms')}")
                    st.write(f"**AI Routed to Nurse:** {result.get('nurse_specialty')}")
                    st.write(f"**AI Measured Vitals:** {result.get('vitals')}")
                    st.write(f"**AI Routed to Doctor:** {result.get('doctor_specialty')}")
                    st.write(f"**AI Generated Prescription:** {result.get('prescription')}")
                
            except Exception as e:
                st.error(f"❌ Error running workflow: {str(e)}")
                st.exception(e)

# Show example symptoms
st.markdown("---")
with st.expander("💡 Example Symptoms to Try"):
    st.markdown("""
    - **Cardiac:** "I have chest pain, shortness of breath, and my heart feels like it's racing"
    - **ENT:** "My ear hurts badly, throat is sore, and I have trouble swallowing"
    - **Orthopedic:** "I twisted my ankle playing basketball and it's very swollen"
    - **Pediatric:** "My 5-year old has high fever, cough, and won't eat anything"
    - **General:** "I feel dizzy and have a headache for the past two days"
    """)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application demonstrates a multi-agent workflow using LangGraph with **LLM-based reasoning** at every step.
    
    **AI-Powered Workflow:**
    1. 👔 Attendant - AI routes patient
    2. 👨‍⚕️ Nurse - AI measures vitals
    3. 👨‍⚕️ Doctor - AI diagnoses & prescribes
    4. 💊 Pharmacist - AI provides instructions
    """)
    
    st.markdown("---")
    st.write("**Tech Stack:**")
    st.code("- Streamlit\n- LangGraph\n- Azure OpenAI\n- Python")
    
    st.markdown("---")
    st.write("**Environment Variables Needed:**")
    st.code("""AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT_NAME""")