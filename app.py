"""
Documentary Production App - Flask backend with Firestore and Vertex AI
"""
import os
import re
import hashlib
import threading
import base64
import uuid
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()  # Load .env file for local development

import requests
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from google.cloud import firestore, storage
from weasyprint import HTML
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, grounding

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    r"https://.*\.run\.app",
])

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-project-id")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", f"{PROJECT_ID}-doc-assets")
MAX_URLS_PER_QUERY = 10
DOWNLOAD_TIMEOUT = 30

# App version and environment
APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")
APP_ENV = os.environ.get("APP_ENV", "prod")

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_NAME)

# Initialize Firestore
db = firestore.Client()

# Initialize Cloud Storage
storage_client = storage.Client()

# Collection prefix based on environment (dev uses separate collections)
COLLECTION_PREFIX = "dev_" if APP_ENV == "dev" else ""

# Collection names (prefixed by environment)
COLLECTIONS = {
    'projects': f'{COLLECTION_PREFIX}doc_projects',
    'episodes': f'{COLLECTION_PREFIX}doc_episodes',
    'series': f'{COLLECTION_PREFIX}doc_series',
    'research': f'{COLLECTION_PREFIX}doc_research',
    'interviews': f'{COLLECTION_PREFIX}doc_interviews',
    'shots': f'{COLLECTION_PREFIX}doc_shots',
    'assets': f'{COLLECTION_PREFIX}doc_assets',
    'scripts': f'{COLLECTION_PREFIX}doc_scripts',
    'feedback': f'{COLLECTION_PREFIX}doc_feedback',
    # Production Factory collections
    'research_documents': f'{COLLECTION_PREFIX}doc_research_documents',
    'archive_logs': f'{COLLECTION_PREFIX}doc_archive_logs',
    'interview_transcripts': f'{COLLECTION_PREFIX}doc_interview_transcripts',
    'script_versions': f'{COLLECTION_PREFIX}doc_script_versions',
    'compliance_items': f'{COLLECTION_PREFIX}doc_compliance_items',
    'agent_tasks': f'{COLLECTION_PREFIX}doc_agent_tasks',
    'users': f'{COLLECTION_PREFIX}doc_users',
}

SEED_USERS = [
    {
        'username': 'Felix',
        'role': 'producer',
        'avatar': 'https://api.dicebear.com/7.x/avataaars/svg?seed=Felix',
        'bio': 'Executive Producer. Focus on Creative Direction & Final Approval.',
        'customInstructions': 'Tone: High-end documentary, Investigative, Cinematic.',
        'gcpProjectId': 'aim-prod-01',
    },
    {
        'username': 'Sarah',
        'role': 'editor',
        'avatar': 'https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah',
        'bio': 'Senior Editor. Focus on Timeline, Pacing, and Visuals.',
        'customInstructions': 'Tone: Fast-paced, rythmic, engaging edits.',
        'gcpProjectId': 'aim-prod-01',
    },
    {
        'username': 'Marcus',
        'role': 'researcher',
        'avatar': 'https://api.dicebear.com/7.x/avataaars/svg?seed=Marcus',
        'bio': 'Lead Researcher. Focus on Fact-Checking & Deep Research.',
        'customInstructions': 'Tone: Academic, precise, highly cited.',
        'gcpProjectId': 'aim-prod-01',
    },
    {
        'username': 'Elena',
        'role': 'legal',
        'avatar': 'https://api.dicebear.com/7.x/avataaars/svg?seed=Elena',
        'bio': 'Legal Counsel. Focus on Clearance & Compliance.',
        'customInstructions': 'Tone: Formal, compliant, risk-averse.',
        'gcpProjectId': 'aim-prod-01',
    },
]

# ============== Production Factory Constants ==============

# Episode workflow phases
EPISODE_PHASES = {
    'research': {'order': 1, 'name': 'Research', 'description': 'Deep research and fact gathering'},
    'archive': {'order': 2, 'name': 'Archive', 'description': 'Archive collection and processing'},
    'script': {'order': 3, 'name': 'Script', 'description': 'Script generation and refinement'},
    'voiceover': {'order': 4, 'name': 'Voiceover', 'description': 'VO generation and QC'},
    'assembly': {'order': 5, 'name': 'Assembly', 'description': 'Quickture final assembly'},
}

# Phase statuses
PHASE_STATUSES = ['pending', 'in_progress', 'review', 'approved', 'rejected']

# Script version types
SCRIPT_VERSION_TYPES = ['V1_initial', 'V2_producer_review', 'V3_interview_additions', 'V4_locked']

# AI Agent types for script generation
AGENT_TYPES = {
    'research_specialist': {
        'name': 'Research Specialist',
        'role': 'Foundation & Fact Accuracy',
        'responsibilities': ['Verify timeline accuracy', 'Ensure technical explanations', 'Flag claims requiring corroboration', 'Suggest narrative structure']
    },
    'archive_specialist': {
        'name': 'Archive Specialist',
        'role': 'Visual Storytelling',
        'responsibilities': ['Match archive to script moments', 'Identify visual sequences', 'Flag missing footage', 'Suggest pacing']
    },
    'interview_producer': {
        'name': 'Interview Producer',
        'role': 'Human Voices & Emotional Beats',
        'responsibilities': ['Extract best soundbites', 'Identify emotional peaks', 'Match interview content to structure', 'Suggest questions for gaps']
    },
    'script_writer': {
        'name': 'Script Writer',
        'role': 'Narrative Construction',
        'responsibilities': ['Build voiceover narrative', 'Structure story arc', 'Write to broadcast standards', 'Match tone to series bible']
    },
    'fact_checker': {
        'name': 'Fact Checker',
        'role': 'Verification & Compliance',
        'responsibilities': ['Cross-reference every claim', 'Flag legal review items', 'Verify dates and names', 'Generate source citations']
    }
}


# ============== Helper Functions ==============

def doc_to_dict(doc):
    """Convert Firestore document to dict with id."""
    if doc.exists:
        data = doc.to_dict()
        data['id'] = doc.id
        return data
    return None


def get_all_docs(collection_name, project_id=None):
    """Get all documents from a collection, optionally filtered by project."""
    collection = db.collection(COLLECTIONS[collection_name])
    if project_id:
        docs = collection.where('projectId', '==', project_id).stream()
    else:
        docs = collection.stream()
    return [doc_to_dict(doc) for doc in docs]


def get_doc(collection_name, doc_id):
    """Get a single document by ID."""
    doc = db.collection(COLLECTIONS[collection_name]).document(doc_id).get()
    return doc_to_dict(doc)


def create_doc(collection_name, data):
    """Create a new document."""
    data['createdAt'] = datetime.utcnow().isoformat()
    data['updatedAt'] = datetime.utcnow().isoformat()
    doc_ref = db.collection(COLLECTIONS[collection_name]).document()
    doc_ref.set(data)
    data['id'] = doc_ref.id
    return data


def update_doc(collection_name, doc_id, data):
    """Update an existing document."""
    data['updatedAt'] = datetime.utcnow().isoformat()
    db.collection(COLLECTIONS[collection_name]).document(doc_id).update(data)
    return get_doc(collection_name, doc_id)


def delete_doc(collection_name, doc_id):
    """Delete a document."""
    db.collection(COLLECTIONS[collection_name]).document(doc_id).delete()
    return True


# ============== Production Factory Helper Functions ==============

def initialize_episode_workflow(episode_id):
    """Initialize workflow phases for a new episode."""
    workflow = {
        'currentPhase': 'research',
        'phases': {}
    }
    for phase_key, phase_info in EPISODE_PHASES.items():
        workflow['phases'][phase_key] = {
            'status': 'pending' if phase_info['order'] > 1 else 'in_progress',
            'startedAt': None,
            'completedAt': None,
            'reviewNotes': [],
            'assignedTo': None
        }
    # Set first phase as in_progress
    workflow['phases']['research']['startedAt'] = datetime.utcnow().isoformat()
    return workflow


def update_episode_phase(episode_id, phase, status, notes=None):
    """Update an episode's workflow phase status."""
    episode = get_doc('episodes', episode_id)
    if not episode:
        return None

    workflow = episode.get('workflow', initialize_episode_workflow(episode_id))

    if phase not in workflow['phases']:
        return None

    workflow['phases'][phase]['status'] = status

    if status == 'in_progress' and not workflow['phases'][phase]['startedAt']:
        workflow['phases'][phase]['startedAt'] = datetime.utcnow().isoformat()
    elif status == 'approved':
        workflow['phases'][phase]['completedAt'] = datetime.utcnow().isoformat()
        # Move to next phase
        phase_order = EPISODE_PHASES[phase]['order']
        for next_phase, info in EPISODE_PHASES.items():
            if info['order'] == phase_order + 1:
                workflow['currentPhase'] = next_phase
                workflow['phases'][next_phase]['status'] = 'in_progress'
                workflow['phases'][next_phase]['startedAt'] = datetime.utcnow().isoformat()
                break

    if notes:
        workflow['phases'][phase]['reviewNotes'].append({
            'text': notes,
            'timestamp': datetime.utcnow().isoformat()
        })

    return update_doc('episodes', episode_id, {'workflow': workflow})


def get_episode_workflow_status(episode_id):
    """Get detailed workflow status for an episode."""
    episode = get_doc('episodes', episode_id)
    if not episode:
        return None

    workflow = episode.get('workflow', {})
    current_phase = workflow.get('currentPhase', 'research')

    # Calculate overall progress
    completed_phases = sum(
        1 for p in workflow.get('phases', {}).values()
        if p.get('status') == 'approved'
    )
    total_phases = len(EPISODE_PHASES)

    return {
        'episodeId': episode_id,
        'currentPhase': current_phase,
        'currentPhaseName': EPISODE_PHASES.get(current_phase, {}).get('name', 'Unknown'),
        'progress': (completed_phases / total_phases) * 100,
        'completedPhases': completed_phases,
        'totalPhases': total_phases,
        'phases': workflow.get('phases', {}),
        'phaseDefinitions': EPISODE_PHASES
    }


def create_episode_with_buckets(data):
    """Create a new episode with initialized workflow and empty buckets."""
    # Initialize workflow
    data['workflow'] = initialize_episode_workflow(None)

    # Initialize buckets (these are logical containers, actual documents stored separately)
    data['researchBucket'] = {
        'uploadedDocuments': [],
        'agentOutputs': [],
        'factCheckSources': [],
        'notebookLMSource': None
    }
    data['archiveBucket'] = {
        'quicktureLogs': [],
        'nasaMetadata': [],
        'interviewTranscripts': [],
        'referenceFootage': []
    }
    data['scriptWorkspace'] = {
        'referenceTemplate': None,
        'currentVersion': None,
        'versionHistory': [],
        'producerFeedback': []
    }
    data['compliancePackage'] = {
        'exifMetadataLogs': [],
        'sourceCitations': [],
        'archiveLicenses': [],
        'legalSignoff': None
    }

    # Episode brief fields
    data['brief'] = data.get('brief', {
        'summary': '',
        'storyBeats': [],
        'targetInterviewees': [],
        'archiveRequirements': [],
        'uniqueAngle': ''
    })

    return create_doc('episodes', data)


def get_docs_by_episode(collection_name, episode_id):
    """Get all documents from a collection filtered by episode."""
    collection = db.collection(COLLECTIONS[collection_name])
    docs = collection.where('episodeId', '==', episode_id).stream()
    return [doc_to_dict(doc) for doc in docs]


def get_docs_by_series(collection_name, series_id):
    """Get all documents from a collection filtered by series."""
    collection = db.collection(COLLECTIONS[collection_name])
    docs = collection.where('seriesId', '==', series_id).stream()
    return [doc_to_dict(doc) for doc in docs]


def create_agent_task(episode_id, agent_type, task_type, input_data):
    """Create a new AI agent task."""
    task_data = {
        'episodeId': episode_id,
        'agentType': agent_type,
        'agentInfo': AGENT_TYPES.get(agent_type, {}),
        'taskType': task_type,
        'status': 'pending',
        'inputData': input_data,
        'outputData': None,
        'startedAt': None,
        'completedAt': None,
        'error': None
    }
    return create_doc('agent_tasks', task_data)


def update_agent_task(task_id, status, output_data=None, error=None):
    """Update an agent task status."""
    update_data = {'status': status}

    if status == 'in_progress':
        update_data['startedAt'] = datetime.utcnow().isoformat()
    elif status in ['completed', 'failed']:
        update_data['completedAt'] = datetime.utcnow().isoformat()

    if output_data:
        update_data['outputData'] = output_data
    if error:
        update_data['error'] = error

    return update_doc('agent_tasks', task_id, update_data)


def get_project_dashboard_stats(project_id):
    """Get dashboard statistics for a project."""
    # Get all series for project
    series_list = get_all_docs('series', project_id)

    # Get all episodes for project
    all_episodes = get_all_docs('episodes', project_id)

    # Calculate phase statistics
    phase_stats = {phase: {'pending': 0, 'in_progress': 0, 'review': 0, 'approved': 0}
                   for phase in EPISODE_PHASES.keys()}

    for episode in all_episodes:
        workflow = episode.get('workflow', {})
        current_phase = workflow.get('currentPhase', 'research')
        phases = workflow.get('phases', {})

        for phase_key, phase_data in phases.items():
            status = phase_data.get('status', 'pending')
            if status in phase_stats[phase_key]:
                phase_stats[phase_key][status] += 1

    # Episodes by series
    episodes_by_series = {}
    for series in series_list:
        series_episodes = [e for e in all_episodes if e.get('seriesId') == series['id']]
        episodes_by_series[series['id']] = {
            'seriesName': series.get('title', 'Unknown'),
            'totalEpisodes': len(series_episodes),
            'completedEpisodes': sum(1 for e in series_episodes
                                     if e.get('workflow', {}).get('currentPhase') == 'assembly'
                                     and e.get('workflow', {}).get('phases', {}).get('assembly', {}).get('status') == 'approved')
        }

    return {
        'projectId': project_id,
        'totalSeries': len(series_list),
        'totalEpisodes': len(all_episodes),
        'phaseStats': phase_stats,
        'episodesBySeries': episodes_by_series,
        'bottlenecks': [
            {'phase': phase, 'count': stats['review']}
            for phase, stats in phase_stats.items()
            if stats['review'] > 0
        ]
    }


# ============== AI Functions ==============

def generate_ai_response(prompt, system_prompt=""):
    """Generate AI response using Vertex AI."""
    try:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI error: {str(e)}"


def generate_grounded_research(prompt, system_prompt=""):
    """Placeholder - AI research functionality disabled in this build."""
    return {
        'text': 'AI Research functionality is not available in this build.',
        'sources': []
    }


# ============== Source Document Functions ==============

def extract_urls(text):
    """Extract URLs from text, limited to MAX_URLS_PER_QUERY."""
    url_pattern = r'https?://[^\s<>\[\]()"\']+'
    urls = re.findall(url_pattern, text)
    cleaned_urls = []
    for url in urls:
        url = url.rstrip('.,;:!?)')
        if url and len(url) > 10:
            cleaned_urls.append(url)
    unique_urls = list(dict.fromkeys(cleaned_urls))
    return unique_urls[:MAX_URLS_PER_QUERY]


def validate_url(url, timeout=3):
    """Check if a URL is accessible (returns True if reachable)."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        return response.status_code < 400
    except:
        # Try GET if HEAD fails (some servers don't support HEAD)
        try:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.close()
            return response.status_code < 400
        except:
            return False


def filter_valid_urls(urls, max_to_check=5):
    """Filter URLs to only include valid, accessible ones (limited to prevent timeout)."""
    valid_urls = []
    checked = 0
    for url in urls:
        if checked >= max_to_check:
            break
        if len(valid_urls) >= 3:  # Stop once we have enough valid URLs
            break
        if validate_url(url):
            valid_urls.append(url)
            print(f"✓ Valid URL: {url[:60]}...")
        else:
            print(f"✗ Invalid URL: {url[:60]}...")
        checked += 1
    return valid_urls


def convert_to_pdf(html_content, url):
    """Convert HTML content to PDF."""
    try:
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        if '<base' not in html_content.lower():
            html_content = html_content.replace(
                '<head>',
                f'<head><base href="{base_url}">',
                1
            )
        html = HTML(string=html_content, base_url=base_url)
        pdf_bytes = html.write_pdf()
        return pdf_bytes
    except Exception as e:
        print(f"PDF conversion error for {url}: {e}")
        return None


def download_and_store(url, bucket_name, project_id, research_id):
    """Download a URL, convert to PDF, and store in GCS bucket."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').split(';')[0].strip()

        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        if path:
            base_filename = path.split('/')[-1]
            base_filename = re.sub(r'[^\w\-.]', '_', base_filename)
        else:
            base_filename = parsed_url.netloc.replace('.', '_')

        if '.' in base_filename:
            base_filename = base_filename.rsplit('.', 1)[0]

        title = base_filename.replace('_', ' ').title()
        if 'html' in content_type:
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', response.text, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()

        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        bucket = storage_client.bucket(bucket_name)

        result = {"url": url, "title": title, "status": "success"}

        if content_type == 'application/pdf':
            blob_path = f"{project_id}/{url_hash}_{base_filename}.pdf"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(response.content, content_type='application/pdf')
            result["gcsPath"] = blob_path
            result["size_bytes"] = len(response.content)
            result["filename"] = f"{base_filename}.pdf"

        elif 'html' in content_type or 'text' in content_type:
            pdf_bytes = convert_to_pdf(response.text, url)
            if pdf_bytes:
                blob_path = f"{project_id}/{url_hash}_{base_filename}.pdf"
                blob = bucket.blob(blob_path)
                blob.upload_from_string(pdf_bytes, content_type='application/pdf')
                result["gcsPath"] = blob_path
                result["size_bytes"] = len(pdf_bytes)
                result["filename"] = f"{base_filename}.pdf"
            else:
                blob_path = f"{project_id}/{url_hash}_{base_filename}.html"
                blob = bucket.blob(blob_path)
                blob.upload_from_string(response.content, content_type='text/html')
                result["gcsPath"] = blob_path
                result["size_bytes"] = len(response.content)
                result["filename"] = f"{base_filename}.html"
        else:
            ext = content_type.split('/')[-1] if '/' in content_type else 'bin'
            blob_path = f"{project_id}/{url_hash}_{base_filename}.{ext}"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(response.content, content_type=content_type)
            result["gcsPath"] = blob_path
            result["size_bytes"] = len(response.content)
            result["filename"] = f"{base_filename}.{ext}"

        if result.get("gcsPath"):
            create_source_document_asset(project_id, research_id, result)

        return result

    except Exception as e:
        return {"url": url, "status": "error", "error": str(e)}


def create_source_document_asset(project_id, research_id, doc_result):
    """Create a Firestore asset entry for a downloaded source document."""
    try:
        asset_data = {
            "projectId": project_id,
            "researchId": research_id,
            "title": doc_result.get("title", "Untitled Document"),
            "type": "Document",
            "source": doc_result.get("url", ""),
            "gcsPath": doc_result.get("gcsPath", ""),
            "status": "Acquired",
            "isSourceDocument": True,
            "sizeBytes": doc_result.get("size_bytes", 0),
            "filename": doc_result.get("filename", ""),
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat()
        }
        doc_ref = db.collection(COLLECTIONS['assets']).document()
        doc_ref.set(asset_data)
        print(f"Created source document asset: {doc_result.get('title')}")
    except Exception as e:
        print(f"Error creating asset: {e}")


def process_source_documents_async(urls, bucket_name, project_id, research_id):
    """Background thread to download and process source documents."""
    print(f"Starting download of {len(urls)} sources for research {research_id}")
    success_count = 0
    error_count = 0
    for url in urls:
        try:
            print(f"Downloading: {url[:80]}...")
            result = download_and_store(url, bucket_name, project_id, research_id)
            if result.get("status") == "error":
                print(f"Failed to download {url}: {result.get('error')}")
                error_count += 1
            else:
                print(f"Successfully downloaded: {result.get('title', url)}")
                success_count += 1
        except Exception as e:
            print(f"Error processing {url}: {e}")
            error_count += 1
    print(f"Download complete: {success_count} success, {error_count} errors")


def ensure_bucket_exists(bucket_name):
    """Create bucket if it doesn't exist."""
    try:
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            bucket = storage_client.create_bucket(bucket_name, location=LOCATION)
        return True
    except Exception as e:
        print(f"Bucket error: {e}")
        return False


# ============== Routes ==============

@app.route("/")
def index():
    """Serve the main app interface from static folder."""
    return send_from_directory('static', 'index.html')


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


# ============== Project Routes ==============

@app.route("/api/projects", methods=["GET"])
def get_projects():
    """Get all projects."""
    projects = get_all_docs('projects')
    return jsonify(projects)


@app.route("/api/projects", methods=["POST"])
def create_project():
    """Create a new project."""
    data = request.get_json()
    project = create_doc('projects', data)
    return jsonify(project), 201


@app.route("/api/projects/<project_id>", methods=["GET"])
def get_project(project_id):
    """Get a single project."""
    project = get_doc('projects', project_id)
    if project:
        return jsonify(project)
    return jsonify({"error": "Project not found"}), 404


@app.route("/api/projects/<project_id>", methods=["PUT"])
def update_project(project_id):
    """Update a project."""
    data = request.get_json()
    project = update_doc('projects', project_id, data)
    return jsonify(project)


@app.route("/api/projects/<project_id>", methods=["DELETE"])
def delete_project(project_id):
    """Delete a project and all related data."""
    # Delete all related data first
    for collection in ['episodes', 'series', 'research', 'interviews', 'shots', 'assets', 'scripts']:
        docs = db.collection(COLLECTIONS[collection]).where('projectId', '==', project_id).stream()
        for doc in docs:
            doc.reference.delete()

    # Delete the project itself
    delete_doc('projects', project_id)
    return jsonify({"success": True})


# ============== Episode Routes ==============

@app.route("/api/projects/<project_id>/episodes", methods=["GET"])
def get_episodes(project_id):
    """Get all episodes for a project."""
    episodes = get_all_docs('episodes', project_id)
    return jsonify(episodes)


@app.route("/api/episodes", methods=["POST"])
def create_episode():
    """Create a new episode."""
    data = request.get_json()
    episode = create_doc('episodes', data)
    return jsonify(episode), 201


@app.route("/api/episodes/<episode_id>", methods=["PUT"])
def update_episode(episode_id):
    """Update an episode."""
    data = request.get_json()
    episode = update_doc('episodes', episode_id, data)
    return jsonify(episode)


@app.route("/api/episodes/<episode_id>", methods=["DELETE"])
def delete_episode(episode_id):
    """Delete an episode."""
    delete_doc('episodes', episode_id)
    return jsonify({"success": True})


# ============== Series Routes ==============

@app.route("/api/projects/<project_id>/series", methods=["GET"])
def get_series(project_id):
    """Get all series for a project."""
    series = get_all_docs('series', project_id)
    # Sort by order field
    series.sort(key=lambda s: s.get('order', 0))
    return jsonify(series)


@app.route("/api/series", methods=["POST"])
def create_series():
    """Create a new series."""
    data = request.get_json()
    series = create_doc('series', data)
    return jsonify(series), 201


@app.route("/api/series/<series_id>", methods=["PUT"])
def update_series(series_id):
    """Update a series."""
    data = request.get_json()
    series = update_doc('series', series_id, data)
    return jsonify(series)


@app.route("/api/series/<series_id>", methods=["DELETE"])
def delete_series(series_id):
    """Delete a series and ungroup its episodes."""
    # Remove seriesId from all episodes in this series
    episodes = db.collection(COLLECTIONS['episodes']).where('seriesId', '==', series_id).stream()
    for ep in episodes:
        ep.reference.update({'seriesId': None, 'updatedAt': datetime.utcnow().isoformat()})

    delete_doc('series', series_id)
    return jsonify({"success": True})


# ============== Research Routes ==============

@app.route("/api/projects/<project_id>/research", methods=["GET"])
def get_research(project_id):
    """Get all research for a project."""
    research = get_all_docs('research', project_id)
    return jsonify(research)


@app.route("/api/research", methods=["POST"])
def create_research():
    """Create a new research item."""
    data = request.get_json()
    research = create_doc('research', data)
    return jsonify(research), 201


@app.route("/api/research/<research_id>", methods=["PUT"])
def update_research(research_id):
    """Update a research item."""
    data = request.get_json()
    research = update_doc('research', research_id, data)
    return jsonify(research)


@app.route("/api/research/<research_id>", methods=["DELETE"])
def delete_research(research_id):
    """Delete a research item."""
    delete_doc('research', research_id)
    return jsonify({"success": True})


# ============== Interview Routes ==============

@app.route("/api/projects/<project_id>/interviews", methods=["GET"])
def get_interviews(project_id):
    """Get all interviews for a project."""
    interviews = get_all_docs('interviews', project_id)
    return jsonify(interviews)


@app.route("/api/interviews", methods=["POST"])
def create_interview():
    """Create a new interview."""
    data = request.get_json()
    interview = create_doc('interviews', data)
    return jsonify(interview), 201


@app.route("/api/interviews/<interview_id>", methods=["PUT"])
def update_interview(interview_id):
    """Update an interview."""
    data = request.get_json()
    interview = update_doc('interviews', interview_id, data)
    return jsonify(interview)


@app.route("/api/interviews/<interview_id>", methods=["DELETE"])
def delete_interview(interview_id):
    """Delete an interview."""
    delete_doc('interviews', interview_id)
    return jsonify({"success": True})


# ============== Shot Routes ==============

@app.route("/api/projects/<project_id>/shots", methods=["GET"])
def get_shots(project_id):
    """Get all shots for a project."""
    shots = get_all_docs('shots', project_id)
    return jsonify(shots)


@app.route("/api/shots", methods=["POST"])
def create_shot():
    """Create a new shot."""
    data = request.get_json()
    shot = create_doc('shots', data)
    return jsonify(shot), 201


@app.route("/api/shots/<shot_id>", methods=["PUT"])
def update_shot(shot_id):
    """Update a shot."""
    data = request.get_json()
    shot = update_doc('shots', shot_id, data)
    return jsonify(shot)


@app.route("/api/shots/<shot_id>", methods=["DELETE"])
def delete_shot(shot_id):
    """Delete a shot."""
    delete_doc('shots', shot_id)
    return jsonify({"success": True})


# ============== Asset Routes ==============

@app.route("/api/projects/<project_id>/assets", methods=["GET"])
def get_assets(project_id):
    """Get all assets for a project."""
    assets = get_all_docs('assets', project_id)
    return jsonify(assets)


@app.route("/api/assets", methods=["POST"])
def create_asset():
    """Create a new asset."""
    data = request.get_json()
    asset = create_doc('assets', data)
    return jsonify(asset), 201


@app.route("/api/assets/<asset_id>", methods=["PUT"])
def update_asset(asset_id):
    """Update an asset."""
    data = request.get_json()
    asset = update_doc('assets', asset_id, data)
    return jsonify(asset)


@app.route("/api/assets/<asset_id>", methods=["DELETE"])
def delete_asset(asset_id):
    """Delete an asset and its GCS file if it exists."""
    # Get asset first to check for GCS file
    asset = get_doc('assets', asset_id)
    if asset and asset.get('gcsPath'):
        try:
            bucket = storage_client.bucket(STORAGE_BUCKET)
            blob = bucket.blob(asset['gcsPath'])
            if blob.exists():
                blob.delete()
        except Exception as e:
            print(f"Error deleting GCS file: {e}")

    delete_doc('assets', asset_id)
    return jsonify({"success": True})


@app.route("/api/assets/upload", methods=["POST"])
def upload_asset_file():
    """Upload a file for an asset and create/update the asset record."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Get form data
    project_id = request.form.get('projectId')
    asset_id = request.form.get('assetId')  # Optional - for updating existing asset
    episode_id = request.form.get('episodeId')  # Optional - for episode research documents
    series_id = request.form.get('seriesId')  # Optional - for series research documents
    is_research_document = request.form.get('isResearchDocument', 'false').lower() == 'true'
    title = request.form.get('title', file.filename)
    asset_type = request.form.get('type', 'Document')
    status = request.form.get('status', 'Acquired')
    notes = request.form.get('notes', '')

    if not project_id:
        return jsonify({"error": "Project ID is required"}), 400

    try:
        # Read file content
        file_content = file.read()
        file_size = len(file_content)

        # Determine content type
        content_type = file.content_type or 'application/octet-stream'

        # Generate safe filename
        original_filename = file.filename
        safe_filename = re.sub(r'[^\w\-.]', '_', original_filename)

        # Generate unique path
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        blob_path = f"assets/{project_id}/{file_hash}_{safe_filename}"

        # Upload to GCS
        ensure_bucket_exists(STORAGE_BUCKET)
        bucket = storage_client.bucket(STORAGE_BUCKET)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(file_content, content_type=content_type)

        print(f"Uploaded asset file: {blob_path} ({file_size} bytes)")

        # Create or update asset document
        asset_data = {
            "projectId": project_id,
            "title": title,
            "type": asset_type,
            "status": status,
            "notes": notes,
            "gcsPath": blob_path,
            "filename": original_filename,
            "mimeType": content_type,
            "sizeBytes": file_size,
            "hasFile": True,
            "isResearchDocument": is_research_document,
            "updatedAt": datetime.utcnow().isoformat()
        }

        # Add optional entity associations for research documents
        if episode_id:
            asset_data["episodeId"] = episode_id
        if series_id:
            asset_data["seriesId"] = series_id

        if asset_id:
            # Update existing asset
            # First delete old file if exists
            existing = get_doc('assets', asset_id)
            if existing and existing.get('gcsPath') and existing['gcsPath'] != blob_path:
                try:
                    old_blob = bucket.blob(existing['gcsPath'])
                    if old_blob.exists():
                        old_blob.delete()
                except:
                    pass

            asset = update_doc('assets', asset_id, asset_data)
        else:
            # Create new asset
            asset_data['createdAt'] = datetime.utcnow().isoformat()
            asset = create_doc('assets', asset_data)

        return jsonify({
            "success": True,
            "asset": asset,
            "gcsPath": blob_path,
            "filename": original_filename,
            "size": file_size
        }), 201

    except Exception as e:
        print(f"Error uploading asset file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/assets/<asset_id>/file", methods=["GET"])
def get_asset_file(asset_id):
    """Download an asset's file using streaming for all sizes."""
    from flask import stream_with_context
    import sys

    print(f"Download request for asset: {asset_id}", file=sys.stderr)

    asset = get_doc('assets', asset_id)
    if not asset:
        print(f"Asset not found: {asset_id}", file=sys.stderr)
        return jsonify({"error": "Asset not found"}), 404

    if not asset.get('gcsPath'):
        print(f"Asset has no gcsPath: {asset_id}", file=sys.stderr)
        return jsonify({"error": "Asset has no file"}), 404

    gcs_path = asset.get('gcsPath')
    print(f"Attempting to access GCS path: {gcs_path}", file=sys.stderr)

    try:
        bucket = storage_client.bucket(STORAGE_BUCKET)
        blob = bucket.blob(gcs_path)

        print(f"Checking if blob exists...", file=sys.stderr)
        if not blob.exists():
            print(f"Blob does not exist: {gcs_path}", file=sys.stderr)
            return jsonify({"error": "File not found in storage"}), 404

        content_type = asset.get('mimeType', 'application/octet-stream')
        filename = asset.get('filename', 'download')

        # Get file size
        print(f"Reloading blob metadata...", file=sys.stderr)
        blob.reload()
        file_size = blob.size

        print(f"Streaming asset file: {gcs_path} ({file_size} bytes)", file=sys.stderr)

        # Stream using GCS range requests - this works within Cloud Run's response limits
        # Each chunk is fetched separately, avoiding memory issues
        chunk_size = 5 * 1024 * 1024  # 5MB chunks for faster streaming

        def generate():
            """Generator that yields file chunks using range requests."""
            offset = 0
            chunk_num = 0
            while offset < file_size:
                end = min(offset + chunk_size, file_size)
                try:
                    # GCS range requests are inclusive on both ends
                    chunk = blob.download_as_bytes(start=offset, end=end - 1)
                    chunk_num += 1
                    if chunk_num % 10 == 0:  # Log every 10 chunks
                        print(f"Chunk {chunk_num}: {offset}-{end} of {file_size} ({100*end/file_size:.1f}%)")
                    yield chunk
                    offset = end
                except Exception as e:
                    print(f"Error downloading chunk at offset {offset}: {e}")
                    raise

            print(f"Stream complete: {chunk_num} chunks, {file_size} bytes total")

        # Use chunked transfer encoding for streaming (no Content-Length)
        # This allows Cloud Run to stream without buffering the entire response
        response = Response(
            stream_with_context(generate()),
            mimetype=content_type,
        )
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        response.headers['X-Content-Length'] = str(file_size)  # Hint for client progress
        return response

    except Exception as e:
        print(f"Error downloading asset file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Chunked upload for large asset files (>32MB)
CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks


@app.route("/api/assets/upload/init", methods=["POST"])
def init_asset_chunked_upload():
    """Initialize a chunked upload session for large asset files."""
    data = request.get_json()
    filename = data.get('filename', 'upload')
    content_type = data.get('contentType', 'application/octet-stream')
    file_size = data.get('fileSize', 0)
    total_chunks = data.get('totalChunks', 1)
    project_id = data.get('projectId')
    title = data.get('title', filename)
    asset_type = data.get('type', 'Document')
    status = data.get('status', 'Acquired')
    notes = data.get('notes', '')

    if not project_id:
        return jsonify({"error": "Project ID is required"}), 400

    # Generate unique upload ID
    upload_id = hashlib.md5(f"{filename}{project_id}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]

    # Generate safe filename and blob path
    safe_filename = re.sub(r'[^\w\-.]', '_', filename)
    blob_path = f"assets/{project_id}/{upload_id}_{safe_filename}"

    print(f"Initialized chunked upload: {upload_id} for {filename} ({file_size} bytes, {total_chunks} chunks)")

    return jsonify({
        "uploadId": upload_id,
        "blobPath": blob_path,
        "totalChunks": total_chunks,
        "projectId": project_id,
        "filename": filename,
        "contentType": content_type,
        "title": title,
        "type": asset_type,
        "status": status,
        "notes": notes
    })


@app.route("/api/assets/upload/chunk/<upload_id>", methods=["POST"])
def upload_asset_chunk(upload_id):
    """Upload a chunk of a large asset file."""
    chunk_index = int(request.form.get('chunkIndex', 0))
    total_chunks = int(request.form.get('totalChunks', 1))
    blob_path = request.form.get('blobPath', '')
    content_type = request.form.get('contentType', 'application/octet-stream')

    if 'chunk' not in request.files:
        return jsonify({"error": "No chunk data"}), 400

    chunk = request.files['chunk']
    chunk_data = chunk.read()

    try:
        ensure_bucket_exists(STORAGE_BUCKET)
        bucket = storage_client.bucket(STORAGE_BUCKET)

        # Store chunk temporarily
        chunk_blob_name = f"uploads/chunks/{upload_id}/chunk_{chunk_index:04d}"
        chunk_blob = bucket.blob(chunk_blob_name)
        chunk_blob.upload_from_string(chunk_data, content_type='application/octet-stream')

        print(f"Uploaded chunk {chunk_index + 1}/{total_chunks} for {upload_id} ({len(chunk_data)} bytes)")

        return jsonify({
            "status": "uploaded",
            "chunkIndex": chunk_index,
            "totalChunks": total_chunks,
            "bytesUploaded": len(chunk_data)
        })

    except Exception as e:
        print(f"Chunk upload error: {e}")
        return jsonify({"error": f"Chunk upload failed: {str(e)}"}), 500


@app.route("/api/assets/upload/complete/<upload_id>", methods=["POST"])
def complete_asset_chunked_upload(upload_id):
    """Complete a chunked upload by combining chunks and creating the asset."""
    data = request.get_json()
    blob_path = data.get('blobPath', '')
    content_type = data.get('contentType', 'application/octet-stream')
    project_id = data.get('projectId')
    filename = data.get('filename', 'upload')
    title = data.get('title', filename)
    asset_type = data.get('type', 'Document')
    status = data.get('status', 'Acquired')
    notes = data.get('notes', '')
    asset_id = data.get('assetId')  # Optional - for updating existing asset

    if not project_id:
        return jsonify({"error": "Project ID is required"}), 400

    try:
        bucket = storage_client.bucket(STORAGE_BUCKET)

        # List all chunks
        chunk_blobs = list(bucket.list_blobs(prefix=f"uploads/chunks/{upload_id}/"))
        chunk_blobs.sort(key=lambda b: b.name)

        if not chunk_blobs:
            return jsonify({"error": "No chunks found for this upload"}), 400

        print(f"Combining {len(chunk_blobs)} chunks for {upload_id}")

        # Combine all chunk data
        combined_data = b''
        for cb in chunk_blobs:
            combined_data += cb.download_as_bytes()

        file_size = len(combined_data)

        # Upload combined file to final location
        final_blob = bucket.blob(blob_path)
        final_blob.upload_from_string(combined_data, content_type=content_type)

        print(f"Combined file uploaded: {blob_path} ({file_size} bytes)")

        # Clean up chunks
        for cb in chunk_blobs:
            cb.delete()

        # Create or update asset document
        asset_data = {
            "projectId": project_id,
            "title": title,
            "type": asset_type,
            "status": status,
            "notes": notes,
            "gcsPath": blob_path,
            "filename": filename,
            "mimeType": content_type,
            "sizeBytes": file_size,
            "hasFile": True,
            "updatedAt": datetime.utcnow().isoformat()
        }

        if asset_id:
            # Update existing asset
            existing = get_doc('assets', asset_id)
            if existing and existing.get('gcsPath') and existing['gcsPath'] != blob_path:
                try:
                    old_blob = bucket.blob(existing['gcsPath'])
                    if old_blob.exists():
                        old_blob.delete()
                except:
                    pass
            asset = update_doc('assets', asset_id, asset_data)
        else:
            asset_data['createdAt'] = datetime.utcnow().isoformat()
            asset = create_doc('assets', asset_data)

        return jsonify({
            "success": True,
            "asset": asset,
            "gcsPath": blob_path,
            "filename": filename,
            "size": file_size
        }), 201

    except Exception as e:
        print(f"Error completing chunked upload: {e}")
        # Try to clean up chunks on error
        try:
            bucket = storage_client.bucket(STORAGE_BUCKET)
            for cb in bucket.list_blobs(prefix=f"uploads/chunks/{upload_id}/"):
                cb.delete()
        except:
            pass
        return jsonify({"error": str(e)}), 500


# ============== Research Documents Query Routes ==============

@app.route("/api/episodes/<episode_id>/research-documents", methods=["GET"])
def get_episode_research_documents(episode_id):
    """Get all research documents for an episode."""
    try:
        docs_ref = db.collection(COLLECTIONS['assets']).where(
            'episodeId', '==', episode_id
        ).where(
            'isResearchDocument', '==', True
        )
        documents = []
        for doc in docs_ref.stream():
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id
            documents.append(doc_data)
        return jsonify(documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/series/<series_id>/research-documents", methods=["GET"])
def get_series_research_documents(series_id):
    """Get all research documents for a series."""
    try:
        docs_ref = db.collection(COLLECTIONS['assets']).where(
            'seriesId', '==', series_id
        ).where(
            'isResearchDocument', '==', True
        )
        documents = []
        for doc in docs_ref.stream():
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id
            documents.append(doc_data)
        return jsonify(documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<project_id>/research-documents", methods=["GET"])
def get_project_research_documents(project_id):
    """Get all research documents for a project (project-level only, not episode/series)."""
    try:
        # Get research documents that are project-level (no episodeId or seriesId)
        docs_ref = db.collection(COLLECTIONS['assets']).where(
            'projectId', '==', project_id
        ).where(
            'isResearchDocument', '==', True
        )
        documents = []
        for doc in docs_ref.stream():
            doc_data = doc.to_dict()
            # Filter to only project-level (no episode or series association)
            if not doc_data.get('episodeId') and not doc_data.get('seriesId'):
                doc_data['id'] = doc.id
                documents.append(doc_data)
        return jsonify(documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<project_id>/all-research-documents", methods=["GET"])
def get_all_project_research_documents(project_id):
    """Get all research documents for a project (including episode and series docs)."""
    try:
        docs_ref = db.collection(COLLECTIONS['assets']).where(
            'projectId', '==', project_id
        ).where(
            'isResearchDocument', '==', True
        )
        documents = []
        for doc in docs_ref.stream():
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id
            documents.append(doc_data)
        return jsonify(documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<project_id>/assets/clear-sources", methods=["DELETE"])
def clear_source_documents(project_id):
    """Delete all source documents for a project."""
    try:
        # Get all source documents
        docs_ref = db.collection(COLLECTIONS['assets']).where(
            'projectId', '==', project_id
        ).where(
            'isSourceDocument', '==', True
        )

        deleted_count = 0
        bucket = storage_client.bucket(STORAGE_BUCKET)

        for doc in docs_ref.stream():
            doc_data = doc.to_dict()
            # Delete GCS file if exists
            if doc_data.get('gcsPath'):
                try:
                    blob = bucket.blob(doc_data['gcsPath'])
                    if blob.exists():
                        blob.delete()
                except Exception as e:
                    print(f"Error deleting GCS file {doc_data['gcsPath']}: {e}")

            # Delete Firestore document
            doc.reference.delete()
            deleted_count += 1

        return jsonify({"success": True, "deleted": deleted_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<project_id>/assets/download-sources", methods=["POST"])
def download_additional_sources(project_id):
    """Download source documents from a list of URLs."""
    data = request.get_json()
    urls = data.get('urls', [])
    research_id = data.get('researchId', datetime.utcnow().strftime("%Y%m%d_%H%M%S"))

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    ensure_bucket_exists(STORAGE_BUCKET)
    results = []
    for url in urls[:5]:  # Max 5 per request
        try:
            result = download_and_store(url, STORAGE_BUCKET, project_id, research_id)
            results.append({
                "url": url,
                "status": "completed" if result.get("status") == "success" else "error",
                "title": result.get("title", ""),
                "filename": result.get("filename", ""),
                "error": result.get("error")
            })
        except Exception as e:
            results.append({"url": url, "status": "error", "error": str(e)})

    return jsonify({"results": results})


@app.route("/api/projects/<project_id>/assets/download-all", methods=["GET"])
def download_all_source_documents(project_id):
    """Download all source documents as a ZIP file."""
    import zipfile
    import io

    try:
        # Get all source documents
        docs_ref = db.collection(COLLECTIONS['assets']).where(
            'projectId', '==', project_id
        ).where(
            'isSourceDocument', '==', True
        )

        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        bucket = storage_client.bucket(STORAGE_BUCKET)

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for doc in docs_ref.stream():
                doc_data = doc.to_dict()
                if doc_data.get('gcsPath'):
                    try:
                        blob = bucket.blob(doc_data['gcsPath'])
                        if blob.exists():
                            content = blob.download_as_bytes()
                            filename = doc_data.get('filename') or doc_data['gcsPath'].split('/')[-1]
                            zip_file.writestr(filename, content)
                    except Exception as e:
                        print(f"Error adding {doc_data['gcsPath']} to ZIP: {e}")

        zip_buffer.seek(0)

        return Response(
            zip_buffer.getvalue(),
            mimetype='application/zip',
            headers={'Content-Disposition': f'attachment; filename="source-documents-{project_id[:8]}.zip"'}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== Script Routes ==============

@app.route("/api/projects/<project_id>/scripts", methods=["GET"])
def get_scripts(project_id):
    """Get all scripts for a project."""
    scripts = get_all_docs('scripts', project_id)
    return jsonify(scripts)


@app.route("/api/scripts", methods=["POST"])
def create_script():
    """Create a new script."""
    data = request.get_json()
    script = create_doc('scripts', data)
    return jsonify(script), 201


@app.route("/api/scripts/<script_id>", methods=["PUT"])
def update_script(script_id):
    """Update a script."""
    data = request.get_json()
    script = update_doc('scripts', script_id, data)
    return jsonify(script)


@app.route("/api/scripts/<script_id>", methods=["DELETE"])
def delete_script(script_id):
    """Delete a script."""
    delete_doc('scripts', script_id)
    return jsonify({"success": True})


# ============== Feedback Routes ==============

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Submit user feedback with optional screenshot."""
    data = request.get_json()

    feedback_text = data.get('text', '')
    feedback_type = data.get('type', 'general')
    screenshot_data = data.get('screenshot')
    project_id = data.get('projectId')
    project_title = data.get('projectTitle')
    current_tab = data.get('currentTab')
    user_agent = data.get('userAgent', '')
    screen_size = data.get('screenSize', '')
    timestamp = data.get('timestamp', datetime.utcnow().isoformat())

    if not feedback_text:
        return jsonify({"error": "Feedback text is required"}), 400

    # Save screenshot to GCS if provided
    screenshot_path = None
    if screenshot_data and screenshot_data.startswith('data:image'):
        try:
            # Extract base64 data
            header, base64_data = screenshot_data.split(',', 1)
            image_data = base64.b64decode(base64_data)

            # Generate unique filename
            feedback_id = str(uuid.uuid4())[:8]
            screenshot_filename = f"feedback/{feedback_id}_screenshot.jpg"

            # Upload to GCS
            ensure_bucket_exists(STORAGE_BUCKET)
            bucket = storage_client.bucket(STORAGE_BUCKET)
            blob = bucket.blob(screenshot_filename)
            blob.upload_from_string(image_data, content_type='image/jpeg')

            screenshot_path = screenshot_filename
            print(f"Feedback screenshot saved: {screenshot_filename}")

        except Exception as e:
            print(f"Error saving feedback screenshot: {e}")
            # Continue without screenshot

    # Create feedback document
    feedback_doc = {
        'text': feedback_text,
        'type': feedback_type,
        'name': data.get('name', 'Anonymous'),
        'version': data.get('version', ''),
        'url': data.get('url', ''),
        'screenshotPath': screenshot_path,
        'projectId': project_id,
        'projectTitle': project_title,
        'currentTab': current_tab,
        'userAgent': user_agent,
        'screenSize': screen_size,
        'status': 'new',
        'response': '',
        'createdAt': timestamp,
        'updatedAt': timestamp
    }

    # Save to Firestore
    doc_ref = db.collection(COLLECTIONS['feedback']).document()
    doc_ref.set(feedback_doc)
    feedback_doc['id'] = doc_ref.id

    print(f"Feedback submitted: [{feedback_type}] {feedback_text[:50]}...")

    return jsonify({
        "success": True,
        "id": doc_ref.id,
        "feedbackId": doc_ref.id,
        "message": "Feedback received"
    }), 201


@app.route("/api/feedback", methods=["GET"])
def get_all_feedback():
    """Get all feedback (admin view)."""
    docs = db.collection(COLLECTIONS['feedback']).order_by(
        'createdAt', direction=firestore.Query.DESCENDING
    ).limit(100).stream()

    feedback_list = []
    for doc in docs:
        fb = doc.to_dict()
        fb['id'] = doc.id
        feedback_list.append(fb)

    return jsonify(feedback_list)


@app.route("/api/feedback/<feedback_id>", methods=["PUT"])
def update_feedback_status(feedback_id):
    """Update a feedback entry (status and admin response)."""
    try:
        data = request.get_json()
        update_data = {
            "updatedAt": datetime.utcnow().isoformat()
        }
        if 'status' in data:
            update_data['status'] = data['status']
        if 'response' in data:
            update_data['response'] = data['response']

        doc_ref = db.collection(COLLECTIONS['feedback']).document(feedback_id)
        doc_ref.update(update_data)
        return jsonify({"success": True})
    except Exception as e:
        print(f"[ERROR] Failed to update feedback: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<project_id>/blueprint-content", methods=["GET"])
def get_blueprint_content(project_id):
    """Get the blueprint markdown content for a project."""
    project = get_doc('projects', project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    # Check if project has blueprint content stored
    blueprint_content = project.get('blueprintContent', '')

    if not blueprint_content and project.get('blueprintFile'):
        # Try to extract content from stored document
        try:
            file_path = project['blueprintFile'].get('path', '')
            if file_path:
                bucket = storage_client.bucket(STORAGE_BUCKET)
                blob = bucket.blob(file_path)
                if blob.exists():
                    content = blob.download_as_text()
                    # If it's HTML (from PDF), try to extract text
                    if content.startswith('<!DOCTYPE') or content.startswith('<html'):
                        blueprint_content = "Blueprint document available for download"
                    else:
                        blueprint_content = content
        except Exception as e:
            print(f"Error loading blueprint content: {e}")

    return jsonify({"content": blueprint_content})


@app.route("/api/ai/generate-script", methods=["POST"])
def ai_generate_script():
    """Generate a Quickture-compatible documentary script for an episode."""
    data = request.get_json()
    episode_id = data.get('episodeId', '')
    episode_title = data.get('episodeTitle', '')
    episode_description = data.get('episodeDescription', '')
    project_title = data.get('projectTitle', '')
    project_description = data.get('projectDescription', '')
    project_style = data.get('projectStyle', '')
    project_id = data.get('projectId', '')
    duration = data.get('duration', '45 minutes')

    # Get research for this episode if available
    research_content = ""
    if episode_id:
        research_docs = db.collection(COLLECTIONS['research']).where(
            'episodeId', '==', episode_id
        ).limit(1).stream()
        for doc in research_docs:
            research_content = doc.to_dict().get('content', '')[:5000]
            break

    system_prompt = f"""You are a professional documentary script writer creating scripts optimized for AI-assisted editing tools like Quickture.

## PROJECT CONTEXT
- Project: {project_title}
- Style: {project_style or 'Documentary'}
- Description: {project_description}

## QUICKTURE-COMPATIBLE SCRIPT FORMAT

Create a detailed documentary script that includes:

1. **HEADER SECTION**
   - Episode title and number
   - Target duration
   - Style/tone notes for editors

2. **SCENE BREAKDOWN** (use this format for each scene):
   ```
   SCENE [NUMBER]: [SCENE TITLE]
   Duration: [estimated time]
   Location: [setting]
   Mood: [emotional tone]

   VISUAL:
   [Description of what we see - B-roll, interviews, graphics]

   AUDIO:
   [Narration, interview soundbites, ambient sound, music cues]

   NARRATION:
   "[Exact narration text if any]"

   INTERVIEW BITES:
   - [Subject name]: "[Key quote or topic to cover]"

   B-ROLL NEEDED:
   - [List of specific shots needed]

   GRAPHICS/TEXT:
   - [Any lower thirds, titles, or info graphics]

   TRANSITION:
   [How this scene connects to the next]
   ```

3. **SHOT LIST SUMMARY**
   - Numbered list of all shots with descriptions
   - Technical notes (wide/close, handheld/tripod, etc.)

4. **INTERVIEW GUIDE**
   - Questions for each subject
   - Key points to cover

5. **MUSIC/SOUND DESIGN NOTES**
   - Mood suggestions per scene
   - Transition audio cues

## REQUIREMENTS
- Be specific and actionable for editors
- Include timing estimates for pacing
- Mark emotional beats and story arc moments
- Note any archival footage or graphics needed
- Keep narration concise and documentary-style
"""

    research_section = f"\n\nRESEARCH AVAILABLE:\n{research_content}" if research_content else ""

    prompt = f"""Create a detailed, Quickture-compatible documentary script for:

Episode: {episode_title}
Description: {episode_description}
Target Duration: {duration}
{research_section}

Generate a comprehensive production script with scene breakdowns, shot lists, narration, and interview guides."""

    result = generate_ai_response(prompt, system_prompt)

    # Save the script
    script_data = {
        'projectId': project_id,
        'episodeId': episode_id,
        'title': f"Script: {episode_title}",
        'content': result,
        'format': 'quickture',
        'duration': duration,
        'status': 'Draft'
    }

    if project_id:
        saved_script = create_doc('scripts', script_data)
        return jsonify({
            "script": result,
            "saved": True,
            "scriptId": saved_script['id']
        })

    return jsonify({"script": result, "saved": False})


# ============== AI Routes ==============

@app.route("/api/ai/research", methods=["POST"])
def ai_research():
    """AI research - disabled in this build."""
    return jsonify({
        "result": "AI Research functionality is not available in this build.",
        "sources": [],
        "disabled": True
    })


def get_research_document_contents(episode_id=None, series_id=None, project_id=None):
    """Fetch and read contents of research documents for context."""
    documents_context = []

    try:
        # Get episode research documents
        if episode_id:
            docs = db.collection(COLLECTIONS['assets']).where('episodeId', '==', episode_id).where('isResearchDocument', '==', True).stream()
            for doc in docs:
                data = doc.to_dict()
                content = read_document_content(data.get('gcsPath'), data.get('mimeType', ''))
                if content:
                    documents_context.append({
                        'source': f"Episode Document: {data.get('title', data.get('filename', 'Unknown'))}",
                        'content': content
                    })

        # Get series research documents
        if series_id:
            docs = db.collection(COLLECTIONS['assets']).where('seriesId', '==', series_id).where('isResearchDocument', '==', True).stream()
            for doc in docs:
                data = doc.to_dict()
                content = read_document_content(data.get('gcsPath'), data.get('mimeType', ''))
                if content:
                    documents_context.append({
                        'source': f"Series Document: {data.get('title', data.get('filename', 'Unknown'))}",
                        'content': content
                    })

        # Get project-level research documents
        if project_id:
            docs = db.collection(COLLECTIONS['assets']).where('projectId', '==', project_id).where('isResearchDocument', '==', True).stream()
            for doc in docs:
                data = doc.to_dict()
                # Only include project-level docs (not linked to episode/series)
                if not data.get('episodeId') and not data.get('seriesId'):
                    content = read_document_content(data.get('gcsPath'), data.get('mimeType', ''))
                    if content:
                        documents_context.append({
                            'source': f"Project Document: {data.get('title', data.get('filename', 'Unknown'))}",
                            'content': content
                        })
    except Exception as e:
        print(f"[ERROR] Error fetching research documents: {e}")

    return documents_context


def read_document_content(gcs_path, mime_type=''):
    """Read content from a document in GCS."""
    if not gcs_path:
        return None

    try:
        bucket = storage_client.bucket(STORAGE_BUCKET)
        blob = bucket.blob(gcs_path)
        content = blob.download_as_bytes()

        # For text-based files, decode to string
        if mime_type.startswith('text/') or gcs_path.endswith(('.txt', '.md', '.csv')):
            return content.decode('utf-8', errors='ignore')[:10000]  # Limit to 10k chars

        # For PDFs and other binary formats, we'd need extraction
        # For now, skip binary files
        if gcs_path.endswith('.pdf'):
            return f"[PDF Document - content extraction not implemented]"

        return None
    except Exception as e:
        print(f"[ERROR] Error reading document {gcs_path}: {e}")
        return None


@app.route("/api/ai/simple-research", methods=["POST"])
def ai_simple_research():
    """AI research query augmented with uploaded research documents from episode and series."""
    data = request.get_json()
    title = data.get('title', '')
    description = data.get('description', '')
    user_query = data.get('query', '')  # User's custom research prompt
    episode_id = data.get('episodeId', '')
    series_id = data.get('seriesId', '')
    project_id = data.get('projectId', '')
    save_research = data.get('save', True)

    print(f"[DEBUG] simple-research called: title={title}, query={user_query[:50] if user_query else 'None'}..., episodeId={episode_id}, seriesId={series_id}")

    # Fetch research documents for context
    research_docs = get_research_document_contents(
        episode_id=episode_id,
        series_id=series_id,
        project_id=project_id
    )

    print(f"[DEBUG] Found {len(research_docs)} research documents for context")

    # Build context from research documents
    context_section = ""
    if research_docs:
        context_section = "\n\n## Reference Documents\n\nThe following research documents have been uploaded and should be used as context:\n\n"
        for doc in research_docs:
            context_section += f"### {doc['source']}\n{doc['content']}\n\n"

    # Use user's query if provided, otherwise fall back to title/description
    research_query = user_query if user_query else f"Research background information for the documentary episode titled '{title}': {description}"

    prompt = f"""You are researching for a documentary episode.

Episode: {title}
{f'Description: {description}' if description else ''}
{context_section}

## Research Request

{research_query}

## Instructions

Based on {'the reference documents above and ' if research_docs else ''}the research request, provide:
- Key facts and background information
- Relevant sources and references (with URLs where possible)
- Interview suggestions (people to talk to)
- Visual/archive material recommendations

{f'IMPORTANT: Incorporate and build upon the information from the {len(research_docs)} provided reference document(s). Reference specific details from them where relevant.' if research_docs else ''}

Format your response with:
- Clear sections with headers
- Bullet points for key facts
- Include real, clickable URLs to credible sources (news sites, Wikipedia, .gov, .edu, .org sites)
- Mark each source with its URL in markdown link format: [Source Name](URL)"""

    system_prompt = """You are a documentary research assistant. Provide comprehensive background research with real source links. Always format URLs as markdown links that can be clicked. Focus on factual, verifiable information from credible sources. When reference documents are provided, incorporate their information and expand upon it."""

    result = generate_ai_response(prompt, system_prompt)

    print(f"[DEBUG] AI response length: {len(result)} chars")

    response_data = {
        "result": result,
        "title": title,
        "query": research_query,
        "saved": False,
        "documentsUsed": len(research_docs)
    }

    # Save research to episode if episodeId provided
    if save_research and episode_id and project_id:
        try:
            print(f"[DEBUG] Saving research to episode {episode_id}")
            # Update the episode with the research content
            episode_ref = db.collection(COLLECTIONS['episodes']).document(episode_id)
            episode_ref.update({
                'research': result,
                'researchGeneratedAt': datetime.utcnow().isoformat(),
                'updatedAt': datetime.utcnow().isoformat()
            })
            response_data['saved'] = True
            response_data['episodeId'] = episode_id
            print(f"[DEBUG] Research saved successfully to episode {episode_id}")
        except Exception as e:
            print(f"[ERROR] Failed to save research: {e}")
            response_data['saveError'] = str(e)

    return jsonify(response_data)


@app.route("/api/episodes/<episode_id>/research", methods=["GET"])
def get_episode_research(episode_id):
    """Get saved research for an episode."""
    print(f"[DEBUG] Getting research for episode {episode_id}")
    try:
        episode = get_doc('episodes', episode_id)
        if not episode:
            print(f"[DEBUG] Episode {episode_id} not found")
            return jsonify({"error": "Episode not found"}), 404

        research = episode.get('research', '')
        generated_at = episode.get('researchGeneratedAt', '')

        print(f"[DEBUG] Found research: {len(research)} chars, generated at: {generated_at}")

        return jsonify({
            "research": research,
            "generatedAt": generated_at,
            "episodeId": episode_id,
            "episodeTitle": episode.get('title', '')
        })
    except Exception as e:
        print(f"[ERROR] Failed to get research: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/episodes/<episode_id>/research", methods=["DELETE"])
def delete_episode_research(episode_id):
    """Delete saved research for an episode."""
    print(f"[DEBUG] Deleting research for episode {episode_id}")
    try:
        episode_ref = db.collection(COLLECTIONS['episodes']).document(episode_id)
        episode_ref.update({
            'research': '',
            'researchGeneratedAt': '',
            'updatedAt': datetime.utcnow().isoformat()
        })
        print(f"[DEBUG] Research deleted for episode {episode_id}")
        return jsonify({"success": True})
    except Exception as e:
        print(f"[ERROR] Failed to delete research: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/episodes/<episode_id>/research", methods=["PUT"])
def save_episode_research(episode_id):
    """Save research to an episode and extract links as reference assets."""
    print(f"[DEBUG] Saving research for episode {episode_id}")
    try:
        data = request.get_json()
        research = data.get('research', '')

        if not research:
            return jsonify({"error": "No research content provided"}), 400

        # Get episode to find projectId
        episode_ref = db.collection(COLLECTIONS['episodes']).document(episode_id)
        episode_doc = episode_ref.get()
        if not episode_doc.exists:
            return jsonify({"error": "Episode not found"}), 404

        episode_data = episode_doc.to_dict()
        project_id = episode_data.get('projectId')
        episode_title = episode_data.get('title', 'Unknown Episode')

        # Save research to episode
        episode_ref.update({
            'research': research,
            'researchGeneratedAt': datetime.utcnow().isoformat(),
            'updatedAt': datetime.utcnow().isoformat()
        })
        print(f"[DEBUG] Research saved for episode {episode_id}, length: {len(research)}")

        # Extract markdown links and create assets as reference links
        links_created = 0
        markdown_links = []

        if project_id:
            # Find all markdown links: [text](url)
            markdown_links = re.findall(r'\[([^\]]+)\]\((https?://[^)]+)\)', research)
            print(f"[DEBUG] Found {len(markdown_links)} links in research")

            for link_text, link_url in markdown_links:
                try:
                    # Check if asset with this URL already exists for this project
                    existing = db.collection(COLLECTIONS['assets']).where(
                        'projectId', '==', project_id
                    ).where(
                        'source', '==', link_url
                    ).limit(1).get()

                    if len(list(existing)) > 0:
                        print(f"[DEBUG] Asset already exists for URL: {link_url[:50]}...")
                        continue

                    # Create asset for this link as a reference (link only, no download)
                    asset_data = {
                        "projectId": project_id,
                        "episodeId": episode_id,
                        "title": link_text[:100],
                        "type": "Reference",
                        "source": link_url,
                        "status": "Identified",
                        "isSourceDocument": False,
                        "isResearchLink": True,
                        "sourceEpisode": episode_title,
                        "notes": f"Extracted from research for: {episode_title}",
                        "createdAt": datetime.utcnow().isoformat(),
                        "updatedAt": datetime.utcnow().isoformat()
                    }
                    doc_ref = db.collection(COLLECTIONS['assets']).document()
                    doc_ref.set(asset_data)
                    links_created += 1
                    print(f"[DEBUG] Created asset: {link_text[:50]}...")
                except Exception as link_error:
                    print(f"[ERROR] Failed to create asset for {link_url}: {link_error}")

        print(f"[DEBUG] Created {links_created} new reference assets")
        return jsonify({
            "success": True,
            "episodeId": episode_id,
            "linksExtracted": len(markdown_links),
            "assetsCreated": links_created
        })
    except Exception as e:
        print(f"[ERROR] Failed to save research: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai/interview-questions", methods=["POST"])
def ai_interview_questions():
    """Generate interview questions."""
    data = request.get_json()
    subject = data.get('subject', '')
    role = data.get('role', '')
    context = data.get('context', '')
    project_title = data.get('projectTitle', '')

    system_prompt = """You are a documentary interview specialist. Generate thoughtful, open-ended questions that elicit detailed stories and insights. Focus on emotional moments, specific details, and unique perspectives."""

    prompt = f"""Generate 8-10 interview questions for {subject}, who is/was a {role}, for a documentary about "{project_title}".
Focus on: {context}.
Make questions specific, open-ended, and designed to get compelling stories."""

    result = generate_ai_response(prompt, system_prompt)
    return jsonify({"result": result})


@app.route("/api/ai/script-outline", methods=["POST"])
def ai_script_outline():
    """Generate script outline."""
    data = request.get_json()
    title = data.get('title', '')
    topic = data.get('topic', '')
    duration = data.get('duration', '45 minutes')
    project_title = data.get('projectTitle', '')

    system_prompt = """You are a documentary scriptwriter. Create detailed episode outlines with clear acts, narrative arcs, and visual storytelling elements."""

    prompt = f"""Create a detailed outline for a documentary episode titled "{title}" ({duration}) about: {topic}.
This is for the documentary "{project_title}".
Include acts, key beats, narrative arc, and suggested visuals."""

    result = generate_ai_response(prompt, system_prompt)
    return jsonify({"result": result})


@app.route("/api/ai/shot-ideas", methods=["POST"])
def ai_shot_ideas():
    """Generate shot ideas."""
    data = request.get_json()
    scene = data.get('scene', '')
    project_title = data.get('projectTitle', '')

    system_prompt = """You are a documentary cinematographer. Suggest creative, visually compelling shot ideas with specific camera movements, angles, and equipment."""

    prompt = f"""Suggest 5-7 creative shots for: {scene}.
This is for "{project_title}".
Include camera angles, movements, equipment needed, and why each shot would be compelling."""

    result = generate_ai_response(prompt, system_prompt)
    return jsonify({"result": result})


@app.route("/api/ai/expand-topic", methods=["POST"])
def ai_expand_topic():
    """Expand on a topic."""
    data = request.get_json()
    topic = data.get('topic', '')
    project_title = data.get('projectTitle', '')

    system_prompt = """You are a documentary story consultant. Help explore topics by suggesting angles, themes, narrative approaches, and key elements to investigate."""

    prompt = f"""Explore potential angles and approaches for covering "{topic}" in the documentary "{project_title}".
Suggest themes, storylines, key questions to answer, and unique perspectives."""

    result = generate_ai_response(prompt, system_prompt)
    return jsonify({"result": result})


@app.route("/api/ai/episode-research", methods=["POST"])
def ai_episode_research():
    """Episode research - disabled in this build."""
    return jsonify({
        "result": "AI Research functionality is not available in this build.",
        "saved": False,
        "sources": [],
        "disabled": True
    })


@app.route("/api/ai/generate-topics", methods=["POST"])
def ai_generate_topics():
    """Generate episode topics from project title and description."""
    data = request.get_json()
    title = data.get('title', '')
    description = data.get('description', '')
    style = data.get('style', '')
    num_topics = data.get('numTopics', 5)

    system_prompt = """You are a documentary series planner. Generate compelling episode topics that would make a cohesive documentary series.

IMPORTANT: Respond ONLY with a JSON array of episode objects. No markdown, no explanation, just valid JSON.

Each episode object must have:
- "title": A compelling episode title (max 60 chars)
- "description": Brief description of what this episode covers (max 150 chars)
- "order": Episode number (1, 2, 3, etc.)

Example response format:
[
  {"title": "Episode Title Here", "description": "What this episode covers", "order": 1},
  {"title": "Another Episode", "description": "Description of content", "order": 2}
]"""

    style_instruction = f"\nStyle/Approach: {style}\nEnsure all episodes match this documentary style." if style else ""

    prompt = f"""Create {num_topics} episode topics for a documentary series:

Title: {title}
Description: {description}{style_instruction}

Generate episode topics that:
1. Cover the subject comprehensively
2. Have a logical narrative flow
3. Each could stand alone but contribute to the whole
4. Are engaging and specific, not generic

Return ONLY the JSON array, no other text."""

    result = generate_ai_response(prompt, system_prompt)

    # Parse the JSON response
    try:
        # Clean up the response - remove markdown code blocks if present
        cleaned = result.strip()
        if cleaned.startswith('```'):
            # Remove markdown code fence
            lines = cleaned.split('\n')
            cleaned = '\n'.join(lines[1:-1] if lines[-1].startswith('```') else lines[1:])

        import json
        topics = json.loads(cleaned)
        return jsonify({"topics": topics})
    except Exception as e:
        # If parsing fails, return the raw result for debugging
        return jsonify({"topics": [], "raw": result, "error": str(e)})


@app.route("/api/upload/init", methods=["POST"])
def init_chunked_upload():
    """Initialize a chunked upload session for large files."""
    data = request.get_json()
    filename = data.get('filename', 'upload')
    content_type = data.get('contentType', 'application/octet-stream')
    file_size = data.get('fileSize', 0)
    total_chunks = data.get('totalChunks', 1)

    # Generate unique upload ID and blob name
    upload_id = hashlib.md5(f"{filename}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    blob_name = f"uploads/{upload_id}.{ext}"

    return jsonify({
        "uploadId": upload_id,
        "gcsUri": f"gs://{STORAGE_BUCKET}/{blob_name}",
        "blobPath": blob_name,
        "totalChunks": total_chunks
    })


@app.route("/api/upload/chunk/<upload_id>", methods=["POST"])
def upload_chunk(upload_id):
    """Upload a chunk of a large file."""
    chunk_index = int(request.form.get('chunkIndex', 0))
    total_chunks = int(request.form.get('totalChunks', 1))
    blob_path = request.form.get('blobPath', '')
    content_type = request.form.get('contentType', 'application/octet-stream')

    if 'chunk' not in request.files:
        return jsonify({"error": "No chunk data"}), 400

    chunk = request.files['chunk']
    chunk_data = chunk.read()

    try:
        ensure_bucket_exists(STORAGE_BUCKET)
        bucket = storage_client.bucket(STORAGE_BUCKET)

        # Store chunk temporarily
        chunk_blob_name = f"uploads/chunks/{upload_id}/chunk_{chunk_index:04d}"
        chunk_blob = bucket.blob(chunk_blob_name)
        chunk_blob.upload_from_string(chunk_data, content_type='application/octet-stream')

        # If this is the last chunk, combine all chunks
        if chunk_index == total_chunks - 1:
            # List all chunks
            chunk_blobs = list(bucket.list_blobs(prefix=f"uploads/chunks/{upload_id}/"))
            chunk_blobs.sort(key=lambda b: b.name)

            # Combine chunks into final file
            final_blob = bucket.blob(blob_path)

            # Use compose for efficiency (works for up to 32 components)
            if len(chunk_blobs) <= 32:
                final_blob.compose(chunk_blobs)
            else:
                # For more than 32 chunks, we need to compose in stages
                # First, combine all chunk data
                combined_data = b''
                for cb in chunk_blobs:
                    combined_data += cb.download_as_bytes()
                final_blob.upload_from_string(combined_data, content_type=content_type)

            # Clean up chunks
            for cb in chunk_blobs:
                cb.delete()

            return jsonify({
                "status": "complete",
                "gcsUri": f"gs://{STORAGE_BUCKET}/{blob_path}",
                "chunkIndex": chunk_index,
                "totalChunks": total_chunks
            })

        return jsonify({
            "status": "uploaded",
            "chunkIndex": chunk_index,
            "totalChunks": total_chunks
        })

    except Exception as e:
        return jsonify({"error": f"Chunk upload failed: {str(e)}"}), 500


@app.route("/api/ai/analyze-blueprint", methods=["POST"])
def ai_analyze_blueprint():
    """Analyze an uploaded document or video to extract project blueprint.

    Supports two modes:
    1. Direct file upload (for small files < 32MB)
    2. GCS URI (for large files uploaded via signed URL)
    """
    import json
    import tempfile
    import mimetypes
    from vertexai.generative_models import Part

    # Check for GCS URI (for large files uploaded directly to GCS)
    gcs_uri = None
    if request.is_json:
        data = request.get_json()
        gcs_uri = data.get('gcsUri')
        num_episodes = int(data.get('numEpisodes', 5))
        filename = data.get('filename', 'video.mp4')
    else:
        num_episodes = int(request.form.get('numEpisodes', 5))
        gcs_uri = request.form.get('gcsUri')
        filename = request.form.get('filename', '')

    # Mode 1: GCS URI provided (large file already in GCS)
    if gcs_uri:
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'mp4'
        mime_type = mimetypes.guess_type(filename)[0] or 'video/mp4'
        file_content = None  # No content to read, using GCS
        print(f"Analyzing from GCS: {gcs_uri}")

    # Mode 2: Direct file upload (small files)
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = file.filename
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        file_content = file.read()
    else:
        return jsonify({"error": "No file uploaded and no gcsUri provided"}), 400

    system_prompt = """You are a documentary production analyst. Analyze the provided content and create a comprehensive project blueprint document.

IMPORTANT: Respond ONLY with a JSON object. No markdown, no explanation, just valid JSON.

The JSON must have:
- "title": A compelling project title (max 80 chars)
- "description": A comprehensive description of what this documentary should cover (max 500 chars)
- "style": The documentary style/approach that fits best (e.g., "investigative journalism", "observational", "personal narrative", "educational", "cinematic")
- "episodes": An array of episode objects, each with "title", "description", and "order"
- "blueprintDocument": A detailed markdown document (1500-2500 words) that serves as the project blueprint, including:
  * Executive Summary
  * Project Overview and Goals
  * Target Audience
  * Visual Style and Tone
  * Key Themes to Explore
  * Production Approach
  * Episode Breakdown with descriptions
  * Editing Approach (detailed section covering):
    - Pacing and rhythm guidelines
    - Transition styles between scenes/segments
    - Use of B-roll and cutaways
    - Music and sound design direction
    - Graphics and text overlay style
    - Color grading/look recommendations
    - Interview editing approach (jump cuts vs continuous, etc.)
    - Narrative structure and story arc editing
  * Potential Challenges and Considerations

Example response format:
{
  "title": "Documentary Title",
  "description": "What this documentary is about...",
  "style": "investigative journalism",
  "episodes": [
    {"title": "Episode 1 Title", "description": "What this episode covers", "order": 1},
    {"title": "Episode 2 Title", "description": "What this episode covers", "order": 2}
  ],
  "blueprintDocument": "# Project Blueprint\\n\\n## Executive Summary\\n..."
}"""

    try:
        # Determine if it's a video or document
        is_video = mime_type.startswith('video/') or ext in ['mp4', 'mov', 'avi', 'mkv', 'webm']
        is_document = ext in ['pdf', 'txt', 'doc', 'docx', 'md'] or mime_type.startswith('text/')

        # Variable to store blueprint file info
        blueprint_file = None
        source_filename = filename

        if is_video:
            # For videos, use GCS URI for Gemini analysis
            bucket = storage_client.bucket(STORAGE_BUCKET)
            temp_blob = None

            if gcs_uri:
                # Large file already in GCS - use provided URI
                video_uri = gcs_uri
                print(f"Using existing GCS file: {video_uri}")
            else:
                # Small file - upload temporarily to GCS
                file_hash = hashlib.md5(file_content).hexdigest()
                temp_blob_name = f"temp_blueprints/{file_hash}.{ext}"
                temp_blob = bucket.blob(temp_blob_name)
                temp_blob.upload_from_string(file_content, content_type=mime_type)
                video_uri = f"gs://{STORAGE_BUCKET}/{temp_blob_name}"
                print(f"Uploaded temp file: {video_uri}")

            prompt = f"""Analyze this video and create a comprehensive documentary project blueprint.
The video is a reference, sample, or outline for a documentary project.

Based on what you see and hear in the video, create:
1. A compelling project title
2. A comprehensive description
3. The appropriate documentary style
4. {num_episodes} episode topics
5. A detailed blueprint document (1000-2000 words) covering all aspects of the project

Return ONLY the JSON object as specified."""

            # Use multimodal model with video
            video_part = Part.from_uri(video_uri, mime_type=mime_type)
            response = model.generate_content([system_prompt, video_part, prompt])
            result = response.text

            # Delete the temporary video file after analysis (only if we uploaded it)
            if temp_blob:
                def cleanup_video():
                    import time
                    time.sleep(30)
                    try:
                        temp_blob.delete()
                    except:
                        pass
                threading.Thread(target=cleanup_video, daemon=True).start()

        elif is_document:
            # For documents, analyze and generate a blueprint document
            if ext == 'pdf':
                # Use Gemini to analyze PDF
                doc_part = Part.from_data(file_content, mime_type='application/pdf')
                prompt = f"""Analyze this PDF document and create a comprehensive documentary project blueprint.
The document contains information about a documentary project or subject matter.

Based on the content, create:
1. A compelling project title
2. A comprehensive description
3. The appropriate documentary style
4. {num_episodes} episode topics
5. A detailed blueprint document (1000-2000 words) covering all aspects of the project

Return ONLY the JSON object as specified."""

                response = model.generate_content([system_prompt, doc_part, prompt])
                result = response.text
            else:
                # For text files, decode and send as text
                try:
                    text_content = file_content.decode('utf-8')
                except:
                    text_content = file_content.decode('latin-1')

                prompt = f"""Analyze this document and create a comprehensive documentary project blueprint.

DOCUMENT CONTENT:
{text_content[:50000]}

Based on the content, create:
1. A compelling project title
2. A comprehensive description
3. The appropriate documentary style
4. {num_episodes} episode topics
5. A detailed blueprint document (1000-2000 words) covering all aspects of the project

Return ONLY the JSON object as specified."""

                result = generate_ai_response(prompt, system_prompt)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        # Parse the JSON response
        cleaned = result.strip()
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            cleaned = '\n'.join(lines[1:-1] if lines[-1].startswith('```') else lines[1:])

        # Clean control characters that break JSON parsing
        import re
        # Remove all ASCII control characters except newline, tab, carriage return
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
        # Also remove extended control characters (0x80-0x9f)
        cleaned = re.sub(r'[\x80-\x9f]', '', cleaned)

        # Fix unescaped newlines inside JSON string values
        # This is a common issue when AI generates long text content
        def fix_json_strings(json_str):
            """Fix newlines inside JSON string values by properly escaping them."""
            result = []
            in_string = False
            escape_next = False

            for char in json_str:
                if escape_next:
                    result.append(char)
                    escape_next = False
                elif char == '\\':
                    result.append(char)
                    escape_next = True
                elif char == '"':
                    result.append(char)
                    in_string = not in_string
                elif char == '\n' and in_string:
                    # Newline inside string - escape it
                    result.append('\\n')
                elif char == '\r' and in_string:
                    # Carriage return inside string - escape it
                    result.append('\\r')
                elif char == '\t' and in_string:
                    # Tab inside string - escape it
                    result.append('\\t')
                else:
                    result.append(char)

            return ''.join(result)

        cleaned = fix_json_strings(cleaned)

        blueprint = json.loads(cleaned)

        # Save the blueprint document to GCS
        blueprint_doc_content = blueprint.get('blueprintDocument', '')
        if not blueprint_doc_content:
            # Generate a basic document if not provided
            blueprint_doc_content = f"""# {blueprint.get('title', 'Documentary Project')}

## Project Overview
{blueprint.get('description', '')}

## Documentary Style
{blueprint.get('style', 'Documentary')}

## Episodes
"""
            for ep in blueprint.get('episodes', []):
                blueprint_doc_content += f"\n### Episode {ep.get('order', '')}: {ep.get('title', '')}\n{ep.get('description', '')}\n"

        # Convert markdown to PDF
        import markdown

        # Convert markdown to HTML
        html_content = markdown.markdown(blueprint_doc_content, extensions=['tables', 'fenced_code'])

        # Create styled HTML document
        styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            color: #333;
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2 {{
            color: #2563eb;
            margin-top: 30px;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #4b5563;
            margin-top: 20px;
        }}
        p {{
            margin-bottom: 12px;
        }}
        ul, ol {{
            margin-bottom: 16px;
            padding-left: 24px;
        }}
        li {{
            margin-bottom: 6px;
        }}
        strong {{
            color: #1a1a1a;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #2563eb;
        }}
        .header h1 {{
            border: none;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #6b7280;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{blueprint.get('title', 'Documentary Blueprint')}</h1>
        <p>Project Blueprint Document</p>
    </div>
    {html_content}
</body>
</html>"""

        # Convert HTML to PDF using WeasyPrint
        pdf_content = HTML(string=styled_html).write_pdf()

        # Save PDF to GCS
        bucket = storage_client.bucket(STORAGE_BUCKET)
        doc_hash = hashlib.md5(blueprint_doc_content.encode()).hexdigest()
        doc_blob_name = f"blueprints/{doc_hash}_blueprint.pdf"
        doc_blob = bucket.blob(doc_blob_name)
        doc_blob.upload_from_string(pdf_content, content_type='application/pdf')

        # Create blueprint file info for the document
        blueprint["blueprintFile"] = {
            "path": doc_blob_name,
            "filename": f"{blueprint.get('title', 'Blueprint')[:50]}_blueprint.pdf",
            "mimeType": "application/pdf",
            "size": len(pdf_content),
            "type": "document",
            "sourceFile": source_filename if is_video else None
        }

        # Include the markdown content for inline display
        blueprint["blueprintContent"] = blueprint_doc_content

        return jsonify({"blueprint": blueprint})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== Document Serving Routes ==============

@app.route("/api/document/<path:blob_path>")
def get_document(blob_path):
    """Serve a document from GCS (inline viewing)."""
    try:
        bucket = storage_client.bucket(STORAGE_BUCKET)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            return jsonify({"error": "Document not found"}), 404

        content = blob.download_as_bytes()
        content_type = blob.content_type or 'application/octet-stream'

        return Response(
            content,
            mimetype=content_type,
            headers={'Content-Disposition': f'inline; filename="{blob_path.split("/")[-1]}"'}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/<path:blob_path>")
def download_document(blob_path):
    """Download a document from GCS (attachment)."""
    try:
        bucket = storage_client.bucket(STORAGE_BUCKET)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            return jsonify({"error": "Document not found"}), 404

        content = blob.download_as_bytes()
        content_type = blob.content_type or 'application/octet-stream'
        filename = blob_path.split("/")[-1]

        return Response(
            content,
            mimetype=content_type,
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<project_id>/source-documents", methods=["GET"])
def get_source_documents(project_id):
    """Get all source documents for a project."""
    try:
        docs_ref = db.collection(COLLECTIONS['assets']).where(
            'projectId', '==', project_id
        ).where(
            'isSourceDocument', '==', True
        )
        documents = []
        for doc in docs_ref.stream():
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id
            documents.append(doc_data)
        return jsonify(documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== Initialize Sample Data ==============

@app.route("/api/init-sample-data", methods=["POST"])
def init_sample_data():
    """Initialize sample data for testing."""
    # Check if project already exists
    existing = list(db.collection(COLLECTIONS['projects']).limit(1).stream())
    if existing:
        return jsonify({"message": "Data already exists", "projectId": existing[0].id})

    # Create sample project
    project_data = {
        'title': 'Apollo 11: Journey to the Moon',
        'description': 'A comprehensive documentary series exploring the historic first moon landing',
        'status': 'In Production'
    }
    project = create_doc('projects', project_data)
    project_id = project['id']

    # Create sample episodes
    episodes = [
        {'projectId': project_id, 'title': 'Episode 1: The Race Begins', 'description': 'Cold War context and the space race', 'status': 'Research', 'duration': '45 min'},
        {'projectId': project_id, 'title': 'Episode 2: Preparation', 'description': 'Training and technical development', 'status': 'Planning', 'duration': '45 min'},
        {'projectId': project_id, 'title': 'Episode 3: Launch', 'description': 'The Saturn V launch and journey to the moon', 'status': 'Planning', 'duration': '45 min'},
        {'projectId': project_id, 'title': 'Episode 4: One Small Step', 'description': 'The lunar landing and moonwalk', 'status': 'Planning', 'duration': '45 min'}
    ]
    for ep in episodes:
        create_doc('episodes', ep)

    # Create sample research
    research_items = [
        {'projectId': project_id, 'title': 'NASA Archives Access', 'content': 'Contact: Dr. Sarah Mitchell at NASA History Office. Available footage includes mission control audio, cockpit recordings, and technical schematics.', 'category': 'Archive'},
        {'projectId': project_id, 'title': 'Cold War Context Research', 'content': 'Key sources: "The Right Stuff" by Tom Wolfe, Smithsonian Air & Space Museum archives, Kennedy Space Center historical records.', 'category': 'Background'}
    ]
    for r in research_items:
        create_doc('research', r)

    # Create sample interviews
    interviews = [
        {'projectId': project_id, 'subject': 'Buzz Aldrin', 'role': 'Lunar Module Pilot', 'status': 'Confirmed', 'questions': 'What were your thoughts during descent?\nDescribe the moment of landing.\nWhat did the lunar surface feel like?', 'notes': 'Available for 2-hour interview in Los Angeles'},
        {'projectId': project_id, 'subject': 'Gene Kranz', 'role': 'Flight Director', 'status': 'Requested', 'questions': 'Describe mission control during landing.\nWhat was the most critical moment?\nHow did the team prepare?', 'notes': 'Contact through NASA public affairs'}
    ]
    for i in interviews:
        create_doc('interviews', i)

    # Create sample shots
    shots = [
        {'projectId': project_id, 'description': 'Kennedy Space Center launch pads - modern day', 'location': 'KSC, Florida', 'equipment': '4K drone, cinema camera', 'status': 'Scheduled', 'shootDate': '2026-03-15'},
        {'projectId': project_id, 'description': 'Saturn V rocket at Space Center Houston', 'location': 'Houston, TX', 'equipment': 'Gimbal, 4K camera', 'status': 'Pending'}
    ]
    for s in shots:
        create_doc('shots', s)

    # Create sample assets
    assets = [
        {'projectId': project_id, 'title': 'Original Mission Control Audio', 'type': 'Audio', 'source': 'NASA Archives', 'status': 'Acquired', 'notes': 'Full 8-day mission audio, needs editing'},
        {'projectId': project_id, 'title': 'Apollo 11 Launch Footage', 'type': 'Video', 'source': 'NASA/CBS News Archives', 'status': 'Licensing', 'notes': 'Multiple camera angles, 16mm film transfer'},
        {'projectId': project_id, 'title': 'Lunar Surface Photos', 'type': 'Image', 'source': 'NASA/Hasselblad', 'status': 'Acquired', 'notes': 'High-res scans of original photos'}
    ]
    for a in assets:
        create_doc('assets', a)

    # Create sample script
    script = {
        'projectId': project_id,
        'title': 'Episode 1 Outline',
        'content': '''ACT 1: COLD WAR CONTEXT
- Sputnik shock (1957)
- Kennedy's moon speech (1961)
- Early Mercury/Gemini programs

ACT 2: THE APOLLO PROGRAM
- Tragedy of Apollo 1
- Technical challenges
- Team assembly

ACT 3: APOLLO 11 CREW
- Armstrong, Aldrin, Collins selection
- Training montage
- Mission objectives'''
    }
    create_doc('scripts', script)

    return jsonify({"message": "Sample data created", "projectId": project_id})


# ============== Production Factory Routes ==============

# --- Project Dashboard ---
@app.route("/api/projects/<project_id>/dashboard", methods=["GET"])
def get_project_dashboard(project_id):
    """Get dashboard statistics for a project."""
    try:
        stats = get_project_dashboard_stats(project_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/<project_id>/workflow-overview", methods=["GET"])
def get_project_workflow_overview(project_id):
    """Get workflow overview for all episodes in a project."""
    try:
        episodes = get_all_docs('episodes', project_id)
        overview = []
        for episode in episodes:
            workflow_status = get_episode_workflow_status(episode['id'])
            if workflow_status:
                overview.append({
                    'episodeId': episode['id'],
                    'episodeTitle': episode.get('title', 'Untitled'),
                    'seriesId': episode.get('seriesId'),
                    **workflow_status
                })
        return jsonify(overview)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Episode Workflow ---
@app.route("/api/episodes/<episode_id>/workflow", methods=["GET"])
def get_episode_workflow(episode_id):
    """Get workflow status for an episode."""
    try:
        status = get_episode_workflow_status(episode_id)
        if not status:
            return jsonify({"error": "Episode not found"}), 404
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/episodes/<episode_id>/workflow/phase", methods=["PUT"])
def update_workflow_phase(episode_id):
    """Update an episode's workflow phase status."""
    try:
        data = request.get_json()
        phase = data.get('phase')
        status = data.get('status')
        notes = data.get('notes')

        if not phase or not status:
            return jsonify({"error": "Phase and status are required"}), 400

        if phase not in EPISODE_PHASES:
            return jsonify({"error": f"Invalid phase. Must be one of: {list(EPISODE_PHASES.keys())}"}), 400

        if status not in PHASE_STATUSES:
            return jsonify({"error": f"Invalid status. Must be one of: {PHASE_STATUSES}"}), 400

        result = update_episode_phase(episode_id, phase, status, notes)
        if not result:
            return jsonify({"error": "Episode not found or update failed"}), 404

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/episodes/factory", methods=["POST"])
def create_episode_factory():
    """Create a new episode with Production Factory structure."""
    try:
        data = request.get_json()
        episode = create_episode_with_buckets(data)
        return jsonify(episode), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Research Documents (Research Bucket) ---
@app.route("/api/episodes/<episode_id>/research-bucket", methods=["GET"])
def get_episode_research_bucket(episode_id):
    """Get all research documents from the research bucket for an episode."""
    try:
        docs = get_docs_by_episode('research_documents', episode_id)
        return jsonify(docs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/research-documents", methods=["POST"])
def create_research_document():
    """Create a new research document."""
    try:
        data = request.get_json()
        # Validate required fields
        if not data.get('episodeId'):
            return jsonify({"error": "episodeId is required"}), 400

        # Set document type
        data['documentType'] = data.get('documentType', 'uploaded')  # uploaded, agent_output, fact_check, notebooklm
        data['confidenceLevel'] = data.get('confidenceLevel', 'unverified')  # verified, probable, requires_confirmation

        doc = create_doc('research_documents', data)
        return jsonify(doc), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/research-documents/<doc_id>", methods=["PUT"])
def update_research_document(doc_id):
    """Update a research document."""
    try:
        data = request.get_json()
        doc = update_doc('research_documents', doc_id, data)
        return jsonify(doc)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/research-documents/<doc_id>", methods=["DELETE"])
def delete_research_document(doc_id):
    """Delete a research document."""
    delete_doc('research_documents', doc_id)
    return jsonify({"success": True})


@app.route("/api/research-documents/upload", methods=["POST"])
def upload_research_document():
    """Upload a file as a research document."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Get form data
    episode_id = request.form.get('episodeId')
    project_id = request.form.get('projectId')
    title = request.form.get('title', file.filename)
    document_type = request.form.get('documentType', 'uploaded')
    confidence_level = request.form.get('confidenceLevel', 'requires_confirmation')
    content = request.form.get('content', '')
    source_url = request.form.get('sourceUrl', '')

    if not episode_id:
        return jsonify({"error": "episodeId is required"}), 400

    try:
        # Read file content
        file_content = file.read()
        file_size = len(file_content)

        # Determine content type
        content_type = file.content_type or 'application/octet-stream'

        # Generate safe filename
        original_filename = file.filename
        safe_filename = re.sub(r'[^\w\-.]', '_', original_filename)

        # Generate unique path
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        blob_path = f"research/{episode_id}/{file_hash}_{safe_filename}"

        # Upload to GCS
        ensure_bucket_exists(STORAGE_BUCKET)
        bucket = storage_client.bucket(STORAGE_BUCKET)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(file_content, content_type=content_type)

        print(f"Uploaded research document: {blob_path} ({file_size} bytes)")

        # Determine file type for categorization
        file_ext = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else ''
        file_category = 'document'
        if file_ext in ['mp4', 'mov', 'avi', 'mkv', 'webm']:
            file_category = 'video'
        elif file_ext in ['mp3', 'wav', 'aiff', 'aac', 'm4a']:
            file_category = 'audio'
        elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'tiff', 'bmp', 'webp']:
            file_category = 'image'
        elif file_ext in ['pdf', 'doc', 'docx', 'txt', 'rtf', 'md']:
            file_category = 'document'

        # Create research document entry with file info
        doc_data = {
            'episodeId': episode_id,
            'projectId': project_id,
            'title': title,
            'content': content or f'Uploaded file: {original_filename}',
            'documentType': document_type,
            'confidenceLevel': confidence_level,
            'sourceUrl': source_url,
            'fileCategory': file_category,
            'gcsPath': blob_path,
            'filename': original_filename,
            'mimeType': content_type,
            'sizeBytes': file_size,
            'hasFile': True,
            'uploadedAt': datetime.utcnow().isoformat()
        }

        doc = create_doc('research_documents', doc_data)

        return jsonify({
            "success": True,
            "researchDocument": doc,
            "gcsPath": blob_path,
            "filename": original_filename,
            "size": file_size,
            "category": file_category
        }), 201

    except Exception as e:
        print(f"Error uploading research document: {e}")
        return jsonify({"error": str(e)}), 500


# --- Archive Logs ---
@app.route("/api/episodes/<episode_id>/archive-logs", methods=["GET"])
def get_episode_archive_logs(episode_id):
    """Get all archive logs for an episode."""
    try:
        logs = get_docs_by_episode('archive_logs', episode_id)
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/archive-logs", methods=["POST"])
def create_archive_log():
    """Create a new archive log entry."""
    try:
        data = request.get_json()
        if not data.get('episodeId'):
            return jsonify({"error": "episodeId is required"}), 400

        # Archive log structure matching Quickture export
        data['source'] = data.get('source', 'manual')  # quickture, nasa_api, getty, manual
        data['clipCount'] = data.get('clipCount', 0)

        log = create_doc('archive_logs', data)
        return jsonify(log), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/archive-logs/import-csv", methods=["POST"])
def import_archive_csv():
    """Import archive log from Quickture CSV export."""
    try:
        data = request.get_json()
        episode_id = data.get('episodeId')
        csv_content = data.get('csvContent')
        source = data.get('source', 'quickture')

        if not episode_id or not csv_content:
            return jsonify({"error": "episodeId and csvContent are required"}), 400

        # Parse CSV content (expecting: Filename, Timecode_In, Timecode_Out, Description, Keywords, Technical_Notes, Getty_ID)
        import csv
        import io

        reader = csv.DictReader(io.StringIO(csv_content))
        clips = []
        for row in reader:
            clips.append({
                'filename': row.get('Filename', ''),
                'timecodeIn': row.get('Timecode_In', ''),
                'timecodeOut': row.get('Timecode_Out', ''),
                'description': row.get('Description', ''),
                'keywords': row.get('Keywords', '').split(',') if row.get('Keywords') else [],
                'technicalNotes': row.get('Technical_Notes', ''),
                'gettyId': row.get('Getty_ID', ''),
                'nasaId': row.get('NASA_ID', ''),
            })

        # Create archive log entry
        log_data = {
            'episodeId': episode_id,
            'source': source,
            'clips': clips,
            'clipCount': len(clips),
            'importedAt': datetime.utcnow().isoformat()
        }

        log = create_doc('archive_logs', log_data)
        return jsonify(log), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/archive-logs/<log_id>", methods=["DELETE"])
def delete_archive_log(log_id):
    """Delete an archive log."""
    delete_doc('archive_logs', log_id)
    return jsonify({"success": True})


@app.route("/api/archive-logs/upload", methods=["POST"])
def upload_archive_file():
    """Upload a file to the archive bucket and create an archive log entry."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Get form data
    episode_id = request.form.get('episodeId')
    project_id = request.form.get('projectId')
    source_type = request.form.get('sourceType', 'upload')  # upload, nasa, getty, pond5
    description = request.form.get('description', '')
    keywords = request.form.get('keywords', '')
    technical_notes = request.form.get('technicalNotes', '')

    if not episode_id:
        return jsonify({"error": "episodeId is required"}), 400

    try:
        # Read file content
        file_content = file.read()
        file_size = len(file_content)

        # Determine content type
        content_type = file.content_type or 'application/octet-stream'

        # Generate safe filename
        original_filename = file.filename
        safe_filename = re.sub(r'[^\w\-.]', '_', original_filename)

        # Generate unique path
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        blob_path = f"archive/{episode_id}/{file_hash}_{safe_filename}"

        # Upload to GCS
        ensure_bucket_exists(STORAGE_BUCKET)
        bucket = storage_client.bucket(STORAGE_BUCKET)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(file_content, content_type=content_type)

        print(f"Uploaded archive file: {blob_path} ({file_size} bytes)")

        # Determine file type for categorization
        file_ext = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else ''
        file_category = 'document'
        if file_ext in ['mp4', 'mov', 'avi', 'mkv', 'webm', 'mxf', 'prores']:
            file_category = 'video'
        elif file_ext in ['mp3', 'wav', 'aiff', 'aac', 'm4a']:
            file_category = 'audio'
        elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'tiff', 'bmp', 'webp']:
            file_category = 'image'
        elif file_ext in ['pdf', 'doc', 'docx', 'txt', 'rtf']:
            file_category = 'document'

        # Create archive log entry with file info
        log_data = {
            'episodeId': episode_id,
            'projectId': project_id,
            'source': source_type,
            'sourceName': original_filename,
            'fileCategory': file_category,
            'clips': [{
                'filename': original_filename,
                'description': description,
                'keywords': [k.strip() for k in keywords.split(',') if k.strip()] if keywords else [],
                'technicalNotes': technical_notes,
                'gcsPath': blob_path,
                'mimeType': content_type,
                'sizeBytes': file_size
            }],
            'clipCount': 1,
            'gcsPath': blob_path,
            'filename': original_filename,
            'mimeType': content_type,
            'sizeBytes': file_size,
            'hasFile': True,
            'uploadedAt': datetime.utcnow().isoformat()
        }

        log = create_doc('archive_logs', log_data)

        return jsonify({
            "success": True,
            "archiveLog": log,
            "gcsPath": blob_path,
            "filename": original_filename,
            "size": file_size,
            "category": file_category
        }), 201

    except Exception as e:
        print(f"Error uploading archive file: {e}")
        return jsonify({"error": str(e)}), 500


# --- Interview Transcripts ---
@app.route("/api/episodes/<episode_id>/transcripts", methods=["GET"])
def get_episode_transcripts(episode_id):
    """Get all interview transcripts for an episode."""
    try:
        transcripts = get_docs_by_episode('interview_transcripts', episode_id)
        return jsonify(transcripts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transcripts", methods=["POST"])
def create_transcript():
    """Create a new interview transcript."""
    try:
        data = request.get_json()
        if not data.get('episodeId'):
            return jsonify({"error": "episodeId is required"}), 400

        # Transcript structure
        data['speakerIdentified'] = data.get('speakerIdentified', False)
        data['timecodesAligned'] = data.get('timecodesAligned', False)
        data['segments'] = data.get('segments', [])  # [{timecode, speaker, text, metadata}]

        transcript = create_doc('interview_transcripts', data)
        return jsonify(transcript), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transcripts/<transcript_id>", methods=["PUT"])
def update_transcript(transcript_id):
    """Update an interview transcript."""
    try:
        data = request.get_json()
        transcript = update_doc('interview_transcripts', transcript_id, data)
        return jsonify(transcript)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transcripts/<transcript_id>", methods=["DELETE"])
def delete_transcript(transcript_id):
    """Delete an interview transcript."""
    delete_doc('interview_transcripts', transcript_id)
    return jsonify({"success": True})


# --- Script Versions ---
@app.route("/api/episodes/<episode_id>/script-versions", methods=["GET"])
def get_episode_script_versions(episode_id):
    """Get all script versions for an episode."""
    try:
        versions = get_docs_by_episode('script_versions', episode_id)
        # Sort by version number
        versions.sort(key=lambda x: x.get('versionNumber', 0))
        return jsonify(versions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/script-versions", methods=["POST"])
def create_script_version():
    """Create a new script version."""
    try:
        data = request.get_json()
        if not data.get('episodeId'):
            return jsonify({"error": "episodeId is required"}), 400

        # Get current version count to auto-increment
        existing = get_docs_by_episode('script_versions', data['episodeId'])
        data['versionNumber'] = len(existing) + 1
        data['versionType'] = data.get('versionType', SCRIPT_VERSION_TYPES[min(len(existing), 3)])
        data['isLocked'] = data.get('isLocked', False)
        data['segments'] = data.get('segments', [])  # [{segmentNumber, title, voiceover, archiveRefs, interviewRefs, genAiVisuals}]

        version = create_doc('script_versions', data)

        # Update episode's script workspace
        episode = get_doc('episodes', data['episodeId'])
        if episode:
            script_workspace = episode.get('scriptWorkspace', {})
            script_workspace['currentVersion'] = version['id']
            if 'versionHistory' not in script_workspace:
                script_workspace['versionHistory'] = []
            script_workspace['versionHistory'].append({
                'versionId': version['id'],
                'versionNumber': data['versionNumber'],
                'versionType': data['versionType'],
                'createdAt': version['createdAt']
            })
            update_doc('episodes', data['episodeId'], {'scriptWorkspace': script_workspace})

        return jsonify(version), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/script-versions/<version_id>", methods=["PUT"])
def update_script_version(version_id):
    """Update a script version."""
    try:
        data = request.get_json()
        version = update_doc('script_versions', version_id, data)
        return jsonify(version)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/script-versions/<version_id>/lock", methods=["POST"])
def lock_script_version(version_id):
    """Lock a script version (V4 final)."""
    try:
        version = update_doc('script_versions', version_id, {
            'isLocked': True,
            'lockedAt': datetime.utcnow().isoformat(),
            'versionType': 'V4_locked'
        })
        return jsonify(version)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Compliance Items ---
@app.route("/api/episodes/<episode_id>/compliance", methods=["GET"])
def get_episode_compliance(episode_id):
    """Get all compliance items for an episode."""
    try:
        items = get_docs_by_episode('compliance_items', episode_id)
        return jsonify(items)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compliance", methods=["POST"])
def create_compliance_item():
    """Create a new compliance item."""
    try:
        data = request.get_json()
        if not data.get('episodeId'):
            return jsonify({"error": "episodeId is required"}), 400

        # Compliance item types: source_citation, archive_license, exif_metadata, legal_signoff
        data['itemType'] = data.get('itemType', 'source_citation')
        data['status'] = data.get('status', 'pending')  # pending, verified, flagged

        item = create_doc('compliance_items', data)
        return jsonify(item), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compliance/<item_id>", methods=["PUT"])
def update_compliance_item(item_id):
    """Update a compliance item."""
    try:
        data = request.get_json()
        item = update_doc('compliance_items', item_id, data)
        return jsonify(item)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/episodes/<episode_id>/compliance/export", methods=["GET"])
def export_compliance_package(episode_id):
    """Export compliance package for an episode."""
    try:
        items = get_docs_by_episode('compliance_items', episode_id)
        episode = get_doc('episodes', episode_id)

        # Group by type
        package = {
            'episodeId': episode_id,
            'episodeTitle': episode.get('title', 'Unknown') if episode else 'Unknown',
            'exportedAt': datetime.utcnow().isoformat(),
            'sourceCitations': [i for i in items if i.get('itemType') == 'source_citation'],
            'archiveLicenses': [i for i in items if i.get('itemType') == 'archive_license'],
            'exifMetadata': [i for i in items if i.get('itemType') == 'exif_metadata'],
            'legalSignoffs': [i for i in items if i.get('itemType') == 'legal_signoff'],
            'summary': {
                'totalItems': len(items),
                'verified': sum(1 for i in items if i.get('status') == 'verified'),
                'pending': sum(1 for i in items if i.get('status') == 'pending'),
                'flagged': sum(1 for i in items if i.get('status') == 'flagged')
            }
        }

        return jsonify(package)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Agent Tasks ---
@app.route("/api/episodes/<episode_id>/agent-tasks", methods=["GET"])
def get_episode_agent_tasks(episode_id):
    """Get all agent tasks for an episode."""
    try:
        tasks = get_docs_by_episode('agent_tasks', episode_id)
        return jsonify(tasks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent-tasks", methods=["POST"])
def create_agent_task_route():
    """Create a new agent task."""
    try:
        data = request.get_json()
        episode_id = data.get('episodeId')
        agent_type = data.get('agentType')
        task_type = data.get('taskType')
        input_data = data.get('inputData', {})

        if not episode_id or not agent_type:
            return jsonify({"error": "episodeId and agentType are required"}), 400

        if agent_type not in AGENT_TYPES:
            return jsonify({"error": f"Invalid agent type. Must be one of: {list(AGENT_TYPES.keys())}"}), 400

        task = create_agent_task(episode_id, agent_type, task_type, input_data)
        return jsonify(task), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent-tasks/<task_id>", methods=["PUT"])
def update_agent_task_route(task_id):
    """Update an agent task."""
    try:
        data = request.get_json()
        status = data.get('status')
        output_data = data.get('outputData')
        error = data.get('error')

        task = update_agent_task(task_id, status, output_data, error)
        return jsonify(task)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- AI Agent Swarm for Script Generation ---
@app.route("/api/ai/script-swarm", methods=["POST"])
def ai_script_swarm():
    """Execute script generation using multi-agent swarm architecture."""
    try:
        data = request.get_json()
        episode_id = data.get('episodeId')
        segment_number = data.get('segmentNumber')  # Optional: generate specific segment only

        if not episode_id:
            return jsonify({"error": "episodeId is required"}), 400

        # Get episode data
        episode = get_doc('episodes', episode_id)
        if not episode:
            return jsonify({"error": "Episode not found"}), 404

        # Get research documents
        research_docs = get_docs_by_episode('research_documents', episode_id)

        # Get archive logs
        archive_logs = get_docs_by_episode('archive_logs', episode_id)

        # Get interview transcripts
        transcripts = get_docs_by_episode('interview_transcripts', episode_id)

        # Get series bible if available
        series_id = episode.get('seriesId')
        series = get_doc('series', series_id) if series_id else None

        # Build context for agents
        research_context = "\n\n".join([
            f"**{doc.get('title', 'Untitled')}**\n{doc.get('content', '')}"
            for doc in research_docs
        ])

        archive_context = "\n".join([
            f"- {clip.get('description', '')} [{clip.get('filename', '')}] TC: {clip.get('timecodeIn', '')}-{clip.get('timecodeOut', '')}"
            for log in archive_logs
            for clip in log.get('clips', [])
        ])

        interview_context = "\n\n".join([
            f"**{t.get('speakerName', 'Unknown Speaker')}**\n" +
            "\n".join([f"[{s.get('timecode', '')}] {s.get('text', '')}" for s in t.get('segments', [])])
            for t in transcripts
        ])

        series_bible = series.get('seriesBible', '') if series else ''
        episode_brief = episode.get('brief', {})

        # Create agent tasks
        tasks_created = []

        # Agent 1: Research Specialist - Analyze research and suggest structure
        research_task = create_agent_task(episode_id, 'research_specialist', 'analyze_research', {
            'episodeBrief': episode_brief,
            'researchContext': research_context[:10000]  # Limit context size
        })
        tasks_created.append(research_task)

        # Agent 2: Archive Specialist - Match archive to story beats
        archive_task = create_agent_task(episode_id, 'archive_specialist', 'match_archive', {
            'episodeBrief': episode_brief,
            'archiveContext': archive_context[:10000]
        })
        tasks_created.append(archive_task)

        # Agent 3: Interview Producer - Extract soundbites
        interview_task = create_agent_task(episode_id, 'interview_producer', 'extract_soundbites', {
            'episodeBrief': episode_brief,
            'interviewContext': interview_context[:10000]
        })
        tasks_created.append(interview_task)

        # Execute agents (in production, this would be async)
        # For now, execute synchronously
        results = {}

        for task in tasks_created:
            agent_type = task['agentType']
            agent_info = AGENT_TYPES[agent_type]
            input_data = task['inputData']

            # Build agent-specific prompt
            system_prompt = f"""You are the {agent_info['name']}, specialized in {agent_info['role']}.

Your responsibilities:
{chr(10).join(f'- {r}' for r in agent_info['responsibilities'])}

Series Guidelines: {series_bible[:2000] if series_bible else 'No series bible available'}

Episode Brief:
- Summary: {episode_brief.get('summary', 'Not provided')}
- Story Beats: {', '.join(episode_brief.get('storyBeats', []))}
- Unique Angle: {episode_brief.get('uniqueAngle', 'Not specified')}"""

            if agent_type == 'research_specialist':
                prompt = f"""Analyze the following research and provide:
1. A verified timeline of key events
2. Technical concepts that need explanation
3. Claims that require interview corroboration
4. Suggested 5-7 segment narrative structure

Research Documents:
{input_data.get('researchContext', 'No research available')}"""

            elif agent_type == 'archive_specialist':
                prompt = f"""Review the available archive footage and provide:
1. Key visual sequences that support the story
2. Footage gaps that need B-roll or Gen AI visuals
3. Suggested archive clips for each story beat
4. Pacing recommendations based on available material

Archive Log:
{input_data.get('archiveContext', 'No archive available')}"""

            elif agent_type == 'interview_producer':
                prompt = f"""Analyze the interview transcripts and provide:
1. Top 10 soundbites ranked by emotional impact
2. Soundbites matched to potential story segments
3. Gaps where additional interviews are needed
4. Suggestions for follow-up questions

Interview Transcripts:
{input_data.get('interviewContext', 'No interviews available')}"""

            # Execute AI call
            update_agent_task(task['id'], 'in_progress')
            try:
                response = generate_ai_response(prompt, system_prompt)
                results[agent_type] = response
                update_agent_task(task['id'], 'completed', {'analysis': response})
            except Exception as e:
                update_agent_task(task['id'], 'failed', error=str(e))
                results[agent_type] = f"Error: {str(e)}"

        # Agent 4: Script Writer - Synthesize all inputs into script
        script_writer_prompt = f"""Based on the analysis from the specialist agents, create a documentary script.

RESEARCH SPECIALIST ANALYSIS:
{results.get('research_specialist', 'Not available')}

ARCHIVE SPECIALIST RECOMMENDATIONS:
{results.get('archive_specialist', 'Not available')}

INTERVIEW PRODUCER SOUNDBITES:
{results.get('interview_producer', 'Not available')}

Create a script with 5-7 segments. For each segment include:
1. SEGMENT TITLE
2. [VOICEOVER] - Narration text
3. [ARCHIVE: description/reference] - Visual callouts
4. [INTERVIEW: speaker - timecode or quote] - Interview bites
5. [GEN AI VISUAL: description] - If needed for missing footage

Write in broadcast documentary style. Target 45 minutes total."""

        script_task = create_agent_task(episode_id, 'script_writer', 'generate_script', {
            'agentInputs': results
        })
        update_agent_task(script_task['id'], 'in_progress')

        try:
            script_response = generate_ai_response(script_writer_prompt,
                f"You are the Script Writer agent. Build voiceover narrative with story arc (setup, complication, resolution). Write to broadcast documentary standards. {series_bible[:1000] if series_bible else ''}")
            update_agent_task(script_task['id'], 'completed', {'script': script_response})

            # Agent 5: Fact Checker - Verify claims
            fact_check_prompt = f"""Review this script and verify all factual claims:

{script_response}

For each major claim, provide:
1. The claim text
2. Source verification (from research documents)
3. Confidence level (verified/probable/requires_confirmation)
4. Any legal concerns to flag"""

            fact_task = create_agent_task(episode_id, 'fact_checker', 'verify_script', {
                'script': script_response
            })
            update_agent_task(fact_task['id'], 'in_progress')

            fact_response = generate_ai_response(fact_check_prompt,
                "You are the Fact Checker agent. Cross-reference every major claim. Flag statements requiring legal review. Generate source citation log.")
            update_agent_task(fact_task['id'], 'completed', {'verification': fact_response})

            # Create script version
            script_version = create_doc('script_versions', {
                'episodeId': episode_id,
                'versionNumber': 1,
                'versionType': 'V1_initial',
                'content': script_response,
                'factCheck': fact_response,
                'agentOutputs': {
                    'research_specialist': results.get('research_specialist'),
                    'archive_specialist': results.get('archive_specialist'),
                    'interview_producer': results.get('interview_producer'),
                    'script_writer': script_response,
                    'fact_checker': fact_response
                },
                'isLocked': False
            })

            return jsonify({
                'success': True,
                'scriptVersionId': script_version['id'],
                'script': script_response,
                'factCheck': fact_response,
                'agentOutputs': results
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai/research-agent", methods=["POST"])
def ai_research_agent():
    """Execute deep research for an episode."""
    try:
        data = request.get_json()
        episode_id = data.get('episodeId')
        episode_brief = data.get('brief', {})

        if not episode_id:
            return jsonify({"error": "episodeId is required"}), 400

        # Create research agent task
        task = create_agent_task(episode_id, 'research_specialist', 'deep_research', {
            'brief': episode_brief
        })

        # Build research prompt
        prompt = f"""Based on this episode brief, generate a comprehensive research package:

Episode Summary: {episode_brief.get('summary', 'Not provided')}
Story Beats: {', '.join(episode_brief.get('storyBeats', []))}
Target Interviewees: {', '.join(episode_brief.get('targetInterviewees', []))}
Archive Requirements: {', '.join(episode_brief.get('archiveRequirements', []))}
Unique Angle: {episode_brief.get('uniqueAngle', 'Not specified')}

Generate:
1. 10-15 specific research questions to investigate
2. Timeline of key events requiring verification
3. Technical concepts requiring explanation
4. Potential interview subjects with their expertise areas
5. Archive footage categories needed
6. List of sources to consult (academic papers, documentaries, books, official records)

Format each section clearly with headers."""

        system_prompt = """You are a documentary research specialist. Generate thorough, factually-grounded research questions and identify key sources.
Focus on verifiable facts and primary sources. Identify potential contradictions or controversies that need investigation."""

        update_agent_task(task['id'], 'in_progress')

        try:
            response = generate_ai_response(prompt, system_prompt)

            # Create research document from output
            research_doc = create_doc('research_documents', {
                'episodeId': episode_id,
                'title': 'AI Research Package',
                'content': response,
                'documentType': 'agent_output',
                'confidenceLevel': 'requires_confirmation',
                'agentTaskId': task['id']
            })

            update_agent_task(task['id'], 'completed', {
                'researchPackage': response,
                'documentId': research_doc['id']
            })

            return jsonify({
                'success': True,
                'taskId': task['id'],
                'documentId': research_doc['id'],
                'researchPackage': response
            })

        except Exception as e:
            update_agent_task(task['id'], 'failed', error=str(e))
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Series-Level Knowledge Base ---
@app.route("/api/series/<series_id>/knowledge-base", methods=["GET"])
def get_series_knowledge_base(series_id):
    """Get series-level knowledge base documents."""
    try:
        series = get_doc('series', series_id)
        if not series:
            return jsonify({"error": "Series not found"}), 404

        knowledge_base = series.get('knowledgeBase', {
            'seriesBible': None,
            'brandGuidelines': None,
            'editorialTone': None,
            'archiveCredentials': [],
            'complianceChecklist': None,
            'researchDocuments': []
        })

        return jsonify(knowledge_base)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/series/<series_id>/knowledge-base", methods=["PUT"])
def update_series_knowledge_base(series_id):
    """Update series-level knowledge base."""
    try:
        data = request.get_json()
        series = get_doc('series', series_id)
        if not series:
            return jsonify({"error": "Series not found"}), 404

        knowledge_base = series.get('knowledgeBase', {})
        knowledge_base.update(data)

        updated = update_doc('series', series_id, {'knowledgeBase': knowledge_base})
        return jsonify(updated)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Bulk Operations ---
@app.route("/api/series/<series_id>/episodes/bulk-create", methods=["POST"])
def bulk_create_episodes(series_id):
    """Bulk create episodes for a series."""
    try:
        data = request.get_json()
        episodes_data = data.get('episodes', [])
        project_id = data.get('projectId')

        if not episodes_data:
            return jsonify({"error": "episodes array is required"}), 400

        created = []
        for ep_data in episodes_data:
            ep_data['seriesId'] = series_id
            ep_data['projectId'] = project_id
            episode = create_episode_with_buckets(ep_data)
            created.append(episode)

        return jsonify({
            'success': True,
            'created': len(created),
            'episodes': created
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Voiceover Generation ---
@app.route("/api/ai/generate-voiceover", methods=["POST"])
def ai_generate_voiceover():
    """Generate voiceover audio from script (placeholder for 11 Labs integration)."""
    try:
        data = request.get_json()
        episode_id = data.get('episodeId')
        script_version_id = data.get('scriptVersionId')
        voice_profile = data.get('voiceProfile', 'default')

        if not episode_id or not script_version_id:
            return jsonify({"error": "episodeId and scriptVersionId are required"}), 400

        # Get script version
        script_version = get_doc('script_versions', script_version_id)
        if not script_version:
            return jsonify({"error": "Script version not found"}), 404

        # Extract VO sections from script
        script_content = script_version.get('content', '')

        # Parse [VOICEOVER] sections
        vo_sections = []
        import re
        vo_pattern = r'\[VOICEOVER\](.*?)(?=\[ARCHIVE|\[INTERVIEW|\[GEN AI|\[VOICEOVER\]|SEGMENT|\Z)'
        matches = re.findall(vo_pattern, script_content, re.DOTALL | re.IGNORECASE)

        for i, match in enumerate(matches):
            vo_text = match.strip().strip('"\'')
            if vo_text:
                vo_sections.append({
                    'segmentIndex': i + 1,
                    'text': vo_text,
                    'estimatedDuration': len(vo_text.split()) / 2.5,  # ~150 words per minute
                    'status': 'pending'
                })

        # Create voiceover task record
        vo_task = create_doc('agent_tasks', {
            'episodeId': episode_id,
            'agentType': 'voiceover_generator',
            'taskType': 'generate_voiceover',
            'status': 'pending',
            'inputData': {
                'scriptVersionId': script_version_id,
                'voiceProfile': voice_profile,
                'sections': vo_sections
            },
            'outputData': None
        })

        # NOTE: In production, this would integrate with 11 Labs API
        # For now, return the parsed VO sections
        return jsonify({
            'success': True,
            'taskId': vo_task['id'],
            'voiceProfile': voice_profile,
            'sections': vo_sections,
            'totalSections': len(vo_sections),
            'estimatedTotalDuration': sum(s['estimatedDuration'] for s in vo_sections),
            'message': 'Voiceover generation queued. 11 Labs integration pending.'
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/episodes/<episode_id>/voiceover-status", methods=["GET"])
def get_voiceover_status(episode_id):
    """Get voiceover generation status for an episode."""
    try:
        tasks = get_docs_by_episode('agent_tasks', episode_id)
        vo_tasks = [t for t in tasks if t.get('agentType') == 'voiceover_generator']

        return jsonify({
            'episodeId': episode_id,
            'tasks': vo_tasks,
            'hasVoiceover': len(vo_tasks) > 0,
            'latestStatus': vo_tasks[-1].get('status') if vo_tasks else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Interview Transcription ---
@app.route("/api/ai/transcribe-interview", methods=["POST"])
def ai_transcribe_interview():
    """Transcribe interview audio using Gemini (placeholder)."""
    try:
        data = request.get_json()
        episode_id = data.get('episodeId')
        interview_id = data.get('interviewId')
        audio_url = data.get('audioUrl')
        speaker_name = data.get('speakerName', 'Unknown Speaker')

        if not episode_id:
            return jsonify({"error": "episodeId is required"}), 400

        # Create transcription task
        task = create_agent_task(episode_id, 'interview_producer', 'transcribe_interview', {
            'interviewId': interview_id,
            'audioUrl': audio_url,
            'speakerName': speaker_name
        })

        # NOTE: In production, this would use Gemini's audio capabilities
        # For demo, create a placeholder transcript structure
        transcript_data = {
            'episodeId': episode_id,
            'interviewId': interview_id,
            'speakerName': speaker_name,
            'speakerIdentified': True,
            'timecodesAligned': False,
            'segments': [],
            'status': 'pending',
            'agentTaskId': task['id'],
            'sourceAudioUrl': audio_url
        }

        transcript = create_doc('interview_transcripts', transcript_data)

        return jsonify({
            'success': True,
            'taskId': task['id'],
            'transcriptId': transcript['id'],
            'message': 'Transcription queued. Gemini audio processing integration pending.'
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transcripts/<transcript_id>/process", methods=["POST"])
def process_transcript(transcript_id):
    """Process transcript to extract soundbites and metadata."""
    try:
        transcript = get_doc('interview_transcripts', transcript_id)
        if not transcript:
            return jsonify({"error": "Transcript not found"}), 404

        # Get full transcript text
        full_text = "\n".join([
            f"[{s.get('timecode', '')}] {s.get('text', '')}"
            for s in transcript.get('segments', [])
        ])

        if not full_text:
            return jsonify({"error": "No transcript content to process"}), 400

        # Use AI to extract soundbites and add metadata
        prompt = f"""Analyze this interview transcript and identify:
1. Top 10 most powerful/emotional soundbites (with timecodes if available)
2. Key factual statements that could support documentary narrative
3. Emotional peaks suitable for voiceover bed
4. Any claims that require fact-checking

Interview with: {transcript.get('speakerName', 'Unknown')}

Transcript:
{full_text[:8000]}

Format each soundbite with:
- Timecode (if available)
- Quote text
- Metadata tag (emotional/factual/technical/anecdote)
- Suggested use in documentary"""

        response = generate_ai_response(prompt,
            "You are an interview producer specialist. Extract the most compelling soundbites for documentary use.")

        # Update transcript with processed data
        update_doc('interview_transcripts', transcript_id, {
            'processed': True,
            'processedAt': datetime.utcnow().isoformat(),
            'aiAnalysis': response
        })

        return jsonify({
            'success': True,
            'transcriptId': transcript_id,
            'analysis': response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- NASA API Integration ---
@app.route("/api/nasa/search", methods=["POST"])
def nasa_search():
    """Search NASA archives for footage and images."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        media_type = data.get('mediaType', 'video')  # video, image, audio
        year_start = data.get('yearStart')
        year_end = data.get('yearEnd')

        if not query:
            return jsonify({"error": "query is required"}), 400

        # NASA Images API
        nasa_api_url = "https://images-api.nasa.gov/search"
        params = {
            'q': query,
            'media_type': media_type,
        }
        if year_start:
            params['year_start'] = year_start
        if year_end:
            params['year_end'] = year_end

        response = requests.get(nasa_api_url, params=params, timeout=30)

        if response.status_code != 200:
            return jsonify({"error": f"NASA API error: {response.status_code}"}), 500

        nasa_data = response.json()
        items = nasa_data.get('collection', {}).get('items', [])

        # Format results
        results = []
        for item in items[:50]:  # Limit to 50 results
            item_data = item.get('data', [{}])[0]
            links = item.get('links', [])
            preview_url = next((l.get('href') for l in links if l.get('rel') == 'preview'), None)

            results.append({
                'nasaId': item_data.get('nasa_id'),
                'title': item_data.get('title'),
                'description': item_data.get('description', '')[:500],
                'dateCreated': item_data.get('date_created'),
                'mediaType': item_data.get('media_type'),
                'keywords': item_data.get('keywords', []),
                'previewUrl': preview_url,
                'center': item_data.get('center')
            })

        return jsonify({
            'success': True,
            'query': query,
            'totalResults': nasa_data.get('collection', {}).get('metadata', {}).get('total_hits', 0),
            'results': results
        })

    except requests.Timeout:
        return jsonify({"error": "NASA API timeout"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nasa/import-to-archive", methods=["POST"])
def nasa_import_to_archive():
    """Import NASA search results to episode archive bucket."""
    try:
        data = request.get_json()
        episode_id = data.get('episodeId')
        nasa_items = data.get('items', [])

        if not episode_id or not nasa_items:
            return jsonify({"error": "episodeId and items are required"}), 400

        # Create archive log from NASA items
        clips = []
        for item in nasa_items:
            clips.append({
                'filename': item.get('nasaId', ''),
                'timecodeIn': '',
                'timecodeOut': '',
                'description': item.get('title', ''),
                'keywords': item.get('keywords', []),
                'technicalNotes': f"NASA Center: {item.get('center', 'Unknown')}",
                'nasaId': item.get('nasaId'),
                'previewUrl': item.get('previewUrl'),
                'dateCreated': item.get('dateCreated')
            })

        log_data = {
            'episodeId': episode_id,
            'source': 'nasa_api',
            'clips': clips,
            'clipCount': len(clips),
            'importedAt': datetime.utcnow().isoformat()
        }

        log = create_doc('archive_logs', log_data)

        return jsonify({
            'success': True,
            'logId': log['id'],
            'importedCount': len(clips)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Episode Brief Management ---
@app.route("/api/episodes/<episode_id>/brief", methods=["GET"])
def get_episode_brief(episode_id):
    """Get episode brief."""
    try:
        episode = get_doc('episodes', episode_id)
        if not episode:
            return jsonify({"error": "Episode not found"}), 404

        return jsonify(episode.get('brief', {
            'summary': '',
            'storyBeats': [],
            'targetInterviewees': [],
            'archiveRequirements': [],
            'uniqueAngle': ''
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/episodes/<episode_id>/brief", methods=["PUT"])
def update_episode_brief(episode_id):
    """Update episode brief."""
    try:
        data = request.get_json()
        episode = get_doc('episodes', episode_id)
        if not episode:
            return jsonify({"error": "Episode not found"}), 404

        brief = episode.get('brief', {})
        brief.update(data)

        updated = update_doc('episodes', episode_id, {'brief': brief})
        return jsonify(updated.get('brief', {}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/episodes/<episode_id>/approve-phase", methods=["POST"])
def approve_episode_phase(episode_id):
    """Approve current phase and advance to next."""
    try:
        data = request.get_json()
        notes = data.get('notes', '')

        episode = get_doc('episodes', episode_id)
        if not episode:
            return jsonify({"error": "Episode not found"}), 404

        workflow = episode.get('workflow', {})
        current_phase = workflow.get('currentPhase', 'research')

        # Update phase to approved
        result = update_episode_phase(episode_id, current_phase, 'approved', notes)

        return jsonify({
            'success': True,
            'previousPhase': current_phase,
            'newPhase': result.get('workflow', {}).get('currentPhase'),
            'episode': result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/episodes/<episode_id>/request-revision", methods=["POST"])
def request_episode_revision(episode_id):
    """Request revision for current phase."""
    try:
        data = request.get_json()
        notes = data.get('notes', '')

        if not notes:
            return jsonify({"error": "Revision notes are required"}), 400

        episode = get_doc('episodes', episode_id)
        if not episode:
            return jsonify({"error": "Episode not found"}), 404

        workflow = episode.get('workflow', {})
        current_phase = workflow.get('currentPhase', 'research')

        # Update phase to rejected with notes
        result = update_episode_phase(episode_id, current_phase, 'rejected', notes)

        return jsonify({
            'success': True,
            'phase': current_phase,
            'status': 'rejected',
            'notes': notes,
            'episode': result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== Documentary Studio Compatibility Endpoints ==============
# Thin shim endpoints matching the request/response shapes the React frontend expects.

@app.route("/api/health")
def api_health():
    """Health check for the Documentary Studio frontend."""
    return jsonify({
        "status": "ok",
        "platform": "Vertex AI (Flask)",
        "project": PROJECT_ID,
        "location": LOCATION,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.route("/api/search-archive", methods=["POST"])
def api_search_archive():
    """Archive search via AI prompt → JSON array of clips with real thumbnails."""
    try:
        data = request.get_json()
        query = data.get("query", "")
        source = data.get("source", "general")

        prompt = f"""You are an archive search specialist. Search the "{source}" archive for documentary footage related to: "{query}".

Based on your knowledge of what would be available in {source}'s archive, provide realistic archive footage results.

Return a JSON array of archive clips. Each clip must have:
- title: string
- duration_seconds: number (30-300)
- visual_description: string
- archive_source: string
- category: string (news/documentary/raw_footage/interview/b-roll)
- year_range: string
- quality: string (HD/SD/4K/Film)
- search_term: string (a 2-4 word image search term for finding a real thumbnail of this clip)

Provide 5-8 relevant results. Return ONLY the JSON array."""

        response_text = generate_ai_response(prompt)
        try:
            import json
            result = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError):
            result = []

        # Fetch real thumbnails from NASA Images API for each result
        for clip in result:
            search_term = clip.pop("search_term", clip.get("title", query))
            try:
                nasa_resp = requests.get(
                    "https://images-api.nasa.gov/search",
                    params={"q": search_term, "media_type": "image", "page_size": 1},
                    timeout=5
                )
                if nasa_resp.status_code == 200:
                    items = nasa_resp.json().get("collection", {}).get("items", [])
                    if items and items[0].get("links"):
                        clip["thumbnail_url"] = items[0]["links"][0].get("href", "")
            except Exception:
                pass
            # If NASA API didn't return an image, try with the main query
            if not clip.get("thumbnail_url"):
                try:
                    nasa_resp2 = requests.get(
                        "https://images-api.nasa.gov/search",
                        params={"q": query, "media_type": "image", "page_size": 1},
                        timeout=5
                    )
                    if nasa_resp2.status_code == 200:
                        items2 = nasa_resp2.json().get("collection", {}).get("items", [])
                        if items2 and items2[0].get("links"):
                            clip["thumbnail_url"] = items2[0]["links"][0].get("href", "")
                except Exception:
                    pass

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nasa/search", methods=["GET"])
def api_nasa_search():
    """Search NASA Images API and return real image/video results with thumbnails."""
    query = request.args.get("q", "")
    media_type = request.args.get("media_type", "image")
    page = request.args.get("page", "1")
    page_size = request.args.get("page_size", "20")
    if not query:
        return jsonify({"items": [], "total": 0})
    try:
        resp = requests.get(
            "https://images-api.nasa.gov/search",
            params={
                "q": query,
                "media_type": media_type,
                "page": page,
                "page_size": page_size,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return jsonify({"error": f"NASA API returned {resp.status_code}"}), 502

        collection = resp.json().get("collection", {})
        items = []
        for item in collection.get("items", []):
            data = item.get("data", [{}])[0]
            links = item.get("links", [])
            thumbnail = links[0].get("href", "") if links else ""
            nasa_id = data.get("nasa_id", "")
            items.append({
                "nasa_id": nasa_id,
                "title": data.get("title", ""),
                "description": (data.get("description", "") or "")[:300],
                "date_created": data.get("date_created", ""),
                "media_type": data.get("media_type", "image"),
                "center": data.get("center", ""),
                "keywords": (data.get("keywords", []) or [])[:5],
                "thumbnail_url": thumbnail,
                "href": item.get("href", ""),
            })

        total_hits = collection.get("metadata", {}).get("total_hits", len(items))
        return jsonify({"items": items, "total": total_hits})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze-clip", methods=["POST"])
def api_analyze_clip():
    """Clip visual analysis → {visual_description, mood, quality_score}."""
    try:
        data = request.get_json()
        clip_title = data.get("clipTitle", "")

        prompt = f"""Provide visual analysis for: "{clip_title}".
Return ONLY valid JSON with these keys:
- visual_description: string
- mood: string
- quality_score: number (0-100)"""

        response_text = generate_ai_response(prompt)
        try:
            import json
            result = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError):
            result = {"visual_description": response_text, "mood": "unknown", "quality_score": 50}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/find-experts", methods=["POST"])
def api_find_experts():
    """Expert discovery → JSON array of experts."""
    try:
        data = request.get_json()
        topic = data.get("topic", "")

        prompt = f"""Find 3 real-world experts on: "{topic}".
Return ONLY a JSON array where each object has:
- name: string
- title: string
- affiliation: string
- relevance: string
- relevance_score: number (0.0-1.0)"""

        response_text = generate_ai_response(prompt)
        try:
            import json
            result = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError):
            result = []
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/refine-beat", methods=["POST"])
def api_refine_beat():
    """Script beat rewrite → {refined: text}."""
    try:
        data = request.get_json()
        current_content = data.get("currentContent", "")
        instruction = data.get("instruction", "")
        context = data.get("context", "")

        prompt = f"""You are a script doctor. Rewrite this beat: "{current_content}"
Instruction: "{instruction}"
{f'Context: {context}' if context else ''}
Return ONLY the rewritten text, no markdown or explanations."""

        response_text = generate_ai_response(prompt)
        return jsonify({"refined": response_text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/index-source", methods=["POST"])
def api_index_source():
    """Source indexing (URL/text/youtube) → analysis JSON."""
    try:
        data = request.get_json()
        source_type = data.get("type", "")
        url = data.get("url", "")
        content = data.get("content", "")
        title = data.get("title", "")

        if source_type in ("url", "youtube") and url:
            kind = "YouTube video URL" if source_type == "youtube" else "web URL"
            prompt = f"""Analyze this {kind}: "{url}"
Provide a comprehensive analysis of what this content likely contains.
Return ONLY valid JSON with:
- title: string (inferred title)
- summary: string (200-300 words)
- key_topics: array of strings (5-8 topics)
- key_facts: array of strings (5-8 facts)
- content_type: string (article/research/news/government/video/etc)
- suggested_questions: array of 3 research questions this source could answer"""

        elif source_type == "text" and content:
            prompt = f"""Analyze this research document/text:

"{content[:15000]}"

Return ONLY valid JSON with:
- title: string (infer a title if not obvious)
- summary: string (200-300 words)
- key_topics: array of strings (5-8 topics)
- key_facts: array of strings (5-8 facts)
- content_type: string (research/transcript/notes/article/etc)
- suggested_questions: array of 3 research questions this text could answer"""
        else:
            return jsonify({"error": "Invalid source type or missing content"}), 400

        response_text = generate_ai_response(prompt)
        try:
            import json
            analysis = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError):
            analysis = {"title": title or url or "Unknown", "summary": response_text}

        return jsonify({
            "status": "indexed",
            "title": analysis.get("title", title or url or "Text Document"),
            "summary": analysis.get("summary", ""),
            "key_topics": analysis.get("key_topics", []),
            "key_facts": analysis.get("key_facts", []),
            "content_type": analysis.get("content_type", source_type),
            "suggested_questions": analysis.get("suggested_questions", [])
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/query-sources", methods=["POST"])
def api_query_sources():
    """Cross-source research → {response, key_facts, source_citations, ...}."""
    try:
        data = request.get_json()
        query = data.get("query", "")
        sources = data.get("sources", [])
        engine = data.get("engine", "google_deep_research")

        source_context = "\n\n---\n\n".join(
            f"[Source {i+1}: {s.get('title', 'Untitled')}]\n{s.get('summary', '')}\nKey Facts: {'; '.join(s.get('key_facts', []))}"
            for i, s in enumerate(sources)
        )

        if engine == "google_deep_research":
            prompt = f"""[DEEP RESEARCH MODE] You are a senior documentary researcher with access to the following indexed sources:

{source_context}

Research Question: "{query}"

Conduct a thorough multi-step analysis:
1. Cross-reference information across all sources
2. Identify corroborating evidence and contradictions
3. Note any gaps in the available information
4. Synthesize findings into a comprehensive response

Return ONLY valid JSON with:
- response: string (detailed 300-500 word answer synthesizing all sources)
- key_facts: array of strings (8-12 specific facts)
- source_citations: array of objects with source_title and relevant_info
- confidence_level: string (high/medium/low)
- follow_up_questions: array of 3 questions for deeper research
- contradictions: array of conflicting information found
- gaps: array of information gaps"""
        else:
            prompt = f"""You are a documentary researcher. Based on these sources:

{source_context}

Question: "{query}"

Return ONLY valid JSON with:
- response: string (comprehensive answer)
- key_facts: array of strings
- source_citations: array of objects with source_title and relevant_info
- follow_up_questions: array of 2-3 questions"""

        response_text = generate_ai_response(prompt)
        try:
            import json
            result = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError):
            result = {"response": response_text, "key_facts": [], "source_citations": []}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """General chat with history → {response: text}."""
    try:
        data = request.get_json()
        message = data.get("message", "")
        system_instruction = data.get("systemInstruction", "")
        history = data.get("history", [])

        # Build conversation context from history
        context_parts = []
        for msg in history:
            role = msg.get("role", "user")
            parts = msg.get("parts", [])
            text = " ".join(p.get("text", "") for p in parts)
            if text:
                context_parts.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")

        conversation_context = "\n".join(context_parts)
        full_prompt = f"""{f'Previous conversation:{chr(10)}{conversation_context}{chr(10)}{chr(10)}' if conversation_context else ''}User: {message}"""

        response_text = generate_ai_response(full_prompt, system_prompt=system_instruction)
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== User Profile Endpoints ==============

@app.route("/api/users", methods=["GET"])
def get_users():
    """Get all user profiles."""
    users = get_all_docs('users')
    return jsonify(users)


@app.route("/api/users/<user_id>", methods=["PUT"])
def update_user(user_id):
    """Update a user profile."""
    data = request.get_json()
    user = update_doc('users', user_id, data)
    return jsonify(user)


@app.route("/api/users/seed", methods=["POST"])
def seed_users():
    """Seed initial user profiles (idempotent — only creates if collection is empty)."""
    existing = list(db.collection(COLLECTIONS['users']).limit(1).stream())
    if existing:
        return jsonify({"message": "Users already exist"})

    created = []
    for user_data in SEED_USERS:
        user = create_doc('users', dict(user_data))
        created.append(user)

    return jsonify(created), 201


# ============== Missing Frontend Endpoints ==============


@app.route("/api/analyze-document", methods=["POST"])
def api_analyze_document():
    """Analyze uploaded document content → structured summary."""
    try:
        data = request.get_json()
        content = data.get("content", "")[:15000]
        file_name = data.get("fileName", "unknown")
        file_type = data.get("fileType", "text/plain")

        if "csv" in file_type.lower() or file_name.lower().endswith(".csv"):
            prompt = f"""Analyze this CSV data from file "{file_name}":

"{content}"

Return ONLY valid JSON with:
- title: string (descriptive title for the data)
- summary: string (200-300 word overview of what the data contains)
- key_topics: array of strings (5-8 topics covered)
- key_facts: array of strings (5-8 notable facts or data points)
- timeline_events: array of objects with date and description (extract any chronological events found in the data)"""
        else:
            prompt = f"""Analyze this document "{file_name}" (type: {file_type}):

"{content}"

Return ONLY valid JSON with:
- title: string (descriptive title)
- summary: string (200-300 word overview)
- key_topics: array of strings (5-8 topics)
- key_facts: array of strings (5-8 notable facts)
- timeline_events: array (empty array for non-temporal documents)"""

        response_text = generate_ai_response(prompt)
        try:
            import json
            analysis = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError):
            analysis = {"title": file_name, "summary": response_text, "key_topics": [], "key_facts": [], "timeline_events": []}

        return jsonify({
            "status": "analyzed",
            "title": analysis.get("title", file_name),
            "summary": analysis.get("summary", ""),
            "key_topics": analysis.get("key_topics", []),
            "key_facts": analysis.get("key_facts", []),
            "timeline_events": analysis.get("timeline_events", [])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/series-structure", methods=["POST"])
def api_series_structure():
    """Generate documentary series structure from premise."""
    try:
        data = request.get_json()
        premise = data.get("premise", "")
        episodes_count = data.get("episodesCount", 3)

        prompt = f"""You are a senior documentary series architect. Design a {episodes_count}-episode documentary series based on this premise:

"{premise}"

Return ONLY valid JSON with:
- episodes: array of {episodes_count} objects, each with:
  - title: string (compelling episode title)
  - research_focus: string (what research is needed for this episode)
  - suggested_engine: string (one of: google_deep_research, academic_search, investigative)
- themes: array of strings (3-5 overarching themes that connect the episodes)"""

        response_text = generate_ai_response(prompt)
        try:
            import json
            result = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError):
            result = {"episodes": [], "themes": []}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-script", methods=["POST"])
def api_generate_script():
    """Multi-agent script generation for a documentary."""
    try:
        data = request.get_json()
        title = data.get("title", "")
        description = data.get("description", "")
        duration = data.get("duration", 30)
        reference_style = data.get("referenceStyle", "cinematic")
        research_context = data.get("researchContext", [])
        archive_context = data.get("archiveContext", [])

        research_summary = "\n".join(
            f"- {r}" if isinstance(r, str) else f"- {r.get('title', '')}: {r.get('summary', '')}"
            for r in research_context
        ) if research_context else "No research context provided."

        archive_summary = "\n".join(
            f"- {a}" if isinstance(a, str) else f"- {a.get('title', '')}: {a.get('visual_description', '')}"
            for a in archive_context
        ) if archive_context else "No archive footage context provided."

        prompt = f"""You are a team of documentary scriptwriters. Generate a detailed script for:

Title: "{title}"
Description: "{description}"
Target Duration: {duration} minutes
Style Reference: {reference_style}

Research Context:
{research_summary}

Available Archive Footage:
{archive_summary}

Return ONLY a valid JSON array where each element is an act/segment with:
- title: string (act/segment title)
- scenes: array of scene objects, each with:
  - title: string (scene title)
  - beats: array of beat objects, each with:
    - type: string (MUST be one of: voice_over, expert, archive, ai_visual)
    - content: string (the actual script text or description)
    - speaker: string (narrator name, interviewee, or empty)
    - topic: string (for expert beats: the interview topic)
    - duration_seconds: number (estimated duration for this beat)

Use these beat types:
- voice_over: for narration, voice-over text
- expert: for interview soundbites, expert commentary
- archive: for B-roll footage, archival clips, stock footage descriptions
- ai_visual: for AI-generated visuals, transitions, title cards, graphics

Aim for 3-5 acts with 2-4 scenes each. Total duration should approximate {duration} minutes."""

        response_text = generate_ai_response(prompt)
        try:
            import json
            result = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError):
            result = []
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interview-plan", methods=["POST"])
def api_interview_plan():
    """Generate interview strategy for a scene/topic."""
    try:
        data = request.get_json()
        scene_context = data.get("sceneContext", "")
        topic = data.get("topic", "")

        prompt = f"""You are an expert documentary interview strategist. Create an interview plan for:

Scene Context: "{scene_context}"
Topic: "{topic}"

Return ONLY valid JSON with:
- ideal_soundbite: string (the ideal 1-2 sentence soundbite you want the interviewee to deliver)
- questions: array of strings (8-12 interview questions, ordered from warm-up to probing, designed to naturally elicit the ideal soundbite)"""

        response_text = generate_ai_response(prompt)
        try:
            import json
            result = json.loads(response_text.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError):
            result = {"ideal_soundbite": "", "questions": []}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-broll", methods=["POST"])
def api_generate_broll():
    """Generate B-roll video description via Vertex AI, with Veo generation when available."""
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        # Generate a detailed visual description for the B-roll
        broll_model = GenerativeModel("gemini-2.0-flash-001")
        response = broll_model.generate_content(
            f"""You are a documentary cinematographer. Create a detailed B-roll shot description for:

"{prompt}"

Return valid JSON with:
- "shot_description": string (2-3 sentences describing the ideal B-roll footage)
- "camera_movement": string (e.g. "slow push-in", "static wide", "tracking shot")
- "duration_seconds": number (suggested duration, 5-30 seconds)
- "visual_style": string (e.g. "cinematic", "observational", "aerial")
- "lighting": string (e.g. "natural daylight", "golden hour", "low-key dramatic")
""",
            generation_config={"response_mime_type": "application/json", "temperature": 0.7}
        )
        description_text = response.candidates[0].content.parts[0].text
        import json
        description = json.loads(description_text)

        # TODO: When Veo is available, generate actual video here:
        # from google.cloud import aiplatform
        # video_response = veo_model.generate_video(prompt=description["shot_description"])
        # video_uri = upload_to_gcs(video_response)

        return jsonify({
            "videoUri": f"https://storage.googleapis.com/{STORAGE_BUCKET}/generated/placeholder-broll.mp4",
            "description": description
        })
    except Exception as e:
        print(f"B-roll generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/gcs/serve", methods=["GET"])
def api_gcs_serve():
    """Proxy a file from GCS so the frontend can display uploaded images/videos."""
    gcs_path = request.args.get("path", "")
    if not gcs_path:
        return jsonify({"error": "Missing path parameter"}), 400
    try:
        bucket = storage_client.bucket(STORAGE_BUCKET)
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            return jsonify({"error": "File not found"}), 404
        content = blob.download_as_bytes()
        content_type = blob.content_type or "application/octet-stream"
        from flask import Response
        return Response(content, content_type=content_type, headers={
            "Cache-Control": "public, max-age=86400"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gcs/buckets", methods=["GET"])
def api_gcs_buckets():
    """List GCS buckets with metadata."""
    try:
        buckets = []
        for bucket in storage_client.list_buckets():
            blobs = list(bucket.list_blobs(max_results=1000))
            file_count = len(blobs)
            size_bytes = sum(b.size or 0 for b in blobs)
            buckets.append({
                "name": bucket.name,
                "region": bucket.location or "us-central1",
                "storageClass": bucket.storage_class or "STANDARD",
                "fileCount": file_count,
                "sizeGb": round(size_bytes / (1024 ** 3), 2)
            })
        return jsonify(buckets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vertex/models", methods=["GET"])
def api_vertex_models():
    """List live Cloud Run services as infrastructure models."""
    try:
        import google.auth
        import google.auth.transport.requests as gauth_requests

        credentials, project = google.auth.default()
        auth_req = gauth_requests.Request()
        credentials.refresh(auth_req)

        # Query Cloud Run Admin API for real service data
        cr_resp = requests.get(
            f"https://run.googleapis.com/v2/projects/{PROJECT_ID}/locations/{LOCATION}/services",
            headers={"Authorization": f"Bearer {credentials.token}"},
            timeout=10
        )
        if cr_resp.status_code != 200:
            raise Exception(f"Cloud Run API returned {cr_resp.status_code}")

        services = cr_resp.json().get("services", [])
        models = []
        for svc in services:
            name = svc.get("name", "").split("/")[-1]
            conditions = svc.get("conditions", [])
            ready = any(c.get("type") == "Ready" and c.get("state") == "CONDITION_SUCCEEDED" for c in conditions)
            latest_rev = svc.get("latestReadyRevision", "").split("/")[-1]
            uri = svc.get("uri", "")

            # Health-check the service to get real latency
            latency_ms = 0
            status = "active" if ready else "error"
            try:
                t0 = datetime.now()
                hc = requests.get(uri, timeout=5, allow_redirects=True)
                latency_ms = int((datetime.now() - t0).total_seconds() * 1000)
                if hc.status_code >= 500:
                    status = "error"
            except Exception:
                status = "error"
                latency_ms = 9999

            models.append({
                "id": name,
                "name": name,
                "version": latest_rev[-12:] if latest_rev else "unknown",
                "status": status,
                "latencyMs": latency_ms,
                "callsPerMin": 0,
                "url": uri
            })
        return jsonify(models)
    except Exception as e:
        # Fallback to curated list if Cloud Run API fails
        models = [
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "version": "v2.0", "status": "active", "latencyMs": 450, "callsPerMin": 60},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "version": "v2.5", "status": "active", "latencyMs": 1200, "callsPerMin": 30},
        ]
        return jsonify(models)


@app.route("/api/cloud/stats", methods=["GET"])
def api_cloud_stats():
    """Live infrastructure stats from Firestore and GCS."""
    try:
        collections = ["projects", "series", "episodes", "scripts",
                        "research", "assets", "interviews", "shots", "users"]
        counts = {}
        for coll_name in collections:
            prefix = ENV_PREFIX if ENV_PREFIX and coll_name != "users" else ""
            full_name = f"{prefix}{coll_name}"
            docs = db.collection(full_name).limit(1000).stream()
            counts[coll_name] = sum(1 for _ in docs)

        # GCS quick summary (just bucket count + total size of primary bucket)
        bucket_count = sum(1 for _ in storage_client.list_buckets())
        primary_size_bytes = 0
        try:
            primary_bucket = storage_client.bucket(STORAGE_BUCKET)
            blobs = list(primary_bucket.list_blobs(max_results=500))
            primary_size_bytes = sum(b.size or 0 for b in blobs)
        except Exception:
            pass

        return jsonify({
            "firestore": counts,
            "totalDocuments": sum(counts.values()),
            "gcsBucketCount": bucket_count,
            "primaryBucketSizeGb": round(primary_size_bytes / (1024 ** 3), 2),
            "primaryBucketFiles": len(blobs) if 'blobs' in dir() else 0,
            "region": LOCATION,
            "project": PROJECT_ID
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vertex/deploy", methods=["POST"])
def api_vertex_deploy():
    """Placeholder for model deployment version bump."""
    try:
        data = request.get_json()
        model_id = data.get("modelId", "")
        current_version = data.get("currentVersion", "v1.0")

        # Increment version: v2.0 → v2.1, v3.5 → v3.6
        parts = current_version.lstrip("v").split(".")
        major = parts[0] if parts else "1"
        minor = int(parts[1]) + 1 if len(parts) > 1 else 1
        new_version = f"v{major}.{minor}"

        return jsonify({"newVersion": new_version})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gcs/purge-cache", methods=["POST"])
def api_gcs_purge_cache():
    """Placeholder for GCS cache purge."""
    try:
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== ElevenLabs Proxy Routes ==============

@app.route("/api/elevenlabs/voices", methods=["POST"])
def proxy_elevenlabs_voices():
    """Proxy ElevenLabs voice list through backend so API key stays server-side."""
    try:
        data = request.get_json()
        user_id = data.get("userId")
        if not user_id:
            return jsonify({"error": "userId required"}), 400
        user_doc = get_doc('users', user_id)
        if not user_doc or not user_doc.get('elevenLabsApiKey'):
            return jsonify({"voices": []}), 200
        api_key = user_doc['elevenLabsApiKey']
        resp = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            timeout=10
        )
        if not resp.ok:
            return jsonify({"voices": []}), 200
        voices_data = resp.json()
        voices = [
            {
                "id": v.get("voice_id"),
                "name": v.get("name"),
                "category": v.get("category", "Generated"),
                "provider": "elevenlabs",
                "preview_url": v.get("preview_url")
            }
            for v in voices_data.get("voices", [])
        ]
        return jsonify({"voices": voices})
    except Exception as e:
        print(f"ElevenLabs voices proxy error: {e}")
        return jsonify({"voices": []}), 200


@app.route("/api/elevenlabs/generate", methods=["POST"])
def proxy_elevenlabs_generate():
    """Proxy ElevenLabs TTS through backend so API key stays server-side.
    Falls back to Google Cloud TTS for demo voices or when no key is configured."""
    try:
        data = request.get_json()
        user_id = data.get("userId")
        voice_id = data.get("voiceId")
        text = data.get("text", "")
        settings = data.get("settings", {})
        if not user_id or not voice_id:
            return jsonify({"error": "userId and voiceId required"}), 400

        # Strip HTML/SSML tags from text for clean TTS input
        import re, uuid
        clean_text = re.sub(r'<[^>]*>', '', text).strip()
        # Strip ElevenLabs intonation tags like [slowly], [pause], etc.
        clean_text = re.sub(r'\[(?:slowly|loudly|whisper|emotional|pause|excited|serious|laugh)\]', '', clean_text, flags=re.IGNORECASE).strip()
        if not clean_text:
            return jsonify({"error": "No text to synthesize"}), 400

        user_doc = get_doc('users', user_id)
        use_elevenlabs = (
            user_doc
            and user_doc.get('elevenLabsApiKey')
            and not voice_id.startswith('demo-')
        )

        if use_elevenlabs:
            # Use ElevenLabs API
            api_key = user_doc['elevenLabsApiKey']
            resp = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "text": clean_text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": settings.get("stability", 0.5),
                        "similarity_boost": settings.get("similarity_boost", 0.75),
                        "style": settings.get("style", 0.0),
                        "use_speaker_boost": settings.get("use_speaker_boost", True)
                    }
                },
                timeout=30
            )
            if not resp.ok:
                error_detail = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {}
                return jsonify({"error": error_detail.get("detail", {}).get("message", "Generation failed")}), 500
            audio_content = resp.content
            content_type = "audio/mpeg"
        else:
            # Fallback: Google Cloud Text-to-Speech
            from google.cloud import texttospeech
            tts_client = texttospeech.TextToSpeechClient()

            # Map demo voice IDs to Google Cloud TTS voice names
            gcp_voice_map = {
                'demo-rachel': ('en-US-Studio-O', texttospeech.SsmlVoiceGender.FEMALE),
                'demo-drew': ('en-US-Studio-M', texttospeech.SsmlVoiceGender.MALE),
                'demo-clyde': ('en-US-Studio-Q', texttospeech.SsmlVoiceGender.MALE),
                'demo-domi': ('en-US-Neural2-F', texttospeech.SsmlVoiceGender.FEMALE),
                'demo-bella': ('en-US-Neural2-C', texttospeech.SsmlVoiceGender.FEMALE),
                'demo-antoni': ('en-US-Neural2-D', texttospeech.SsmlVoiceGender.MALE),
            }
            voice_name, gender = gcp_voice_map.get(voice_id, ('en-US-Studio-M', texttospeech.SsmlVoiceGender.MALE))

            synthesis_input = texttospeech.SynthesisInput(text=clean_text)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name,
                ssml_gender=gender
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.95,
                pitch=0.0,
            )
            tts_resp = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice_params, audio_config=audio_config
            )
            audio_content = tts_resp.audio_content
            content_type = "audio/mpeg"

        # Upload audio to GCS and return URL
        audio_filename = f"voiceover/{uuid.uuid4().hex}.mp3"
        bucket = storage_client.bucket(STORAGE_BUCKET)
        blob = bucket.blob(audio_filename)
        blob.upload_from_string(audio_content, content_type=content_type)
        blob.make_public()
        audio_url = blob.public_url
        return jsonify({"audioUrl": audio_url})
    except Exception as e:
        print(f"ElevenLabs/TTS generate proxy error: {e}")
        return jsonify({"error": str(e)}), 500


# ============== Auto-seed users on startup ==============
try:
    if not list(db.collection(COLLECTIONS['users']).limit(1).stream()):
        for _ud in SEED_USERS:
            create_doc('users', dict(_ud))
        print(f"[startup] Seeded {len(SEED_USERS)} default users into {COLLECTIONS['users']}")
    else:
        print(f"[startup] Users already exist in {COLLECTIONS['users']}")
except Exception as _e:
    print(f"[startup] Could not auto-seed users: {_e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
