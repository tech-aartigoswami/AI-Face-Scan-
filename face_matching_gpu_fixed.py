import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from tqdm import tqdm


# ==================================================
# 1. CONFIGURATION
# ==================================================

# Change this path to the folder that contains Folder-A and Folder-B.
BASE_DIR = Path(r"C:\Users\kiran\Desktop\AI Face scan")

FOLDER_A = BASE_DIR / "Folder-A"  # Known person's reference photos
FOLDER_B = BASE_DIR / "Folder-B"  # Candidate photos to scan
FOLDER_C = BASE_DIR / "Folder-C"  # Similarity >= threshold
FOLDER_D = BASE_DIR / "Folder-D"  # Similarity < threshold

THRESHOLD = 0.70

# First run with True to preview. Change to False only after checking the CSV.
DRY_RUN = False

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

REPORT_FILE = BASE_DIR / "face_matching_report.csv"
DETECTION_SIZE = (640, 640)

app = None


# ==================================================
# 2. FOLDER VALIDATION
# ==================================================

def validate_folders():
    if not BASE_DIR.is_dir():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    if not FOLDER_A.is_dir():
        raise FileNotFoundError(f"Folder-A not found: {FOLDER_A}")

    if not FOLDER_B.is_dir():
        raise FileNotFoundError(f"Folder-B not found: {FOLDER_B}")

    folders = {
        "Folder-A": FOLDER_A.resolve(),
        "Folder-B": FOLDER_B.resolve(),
        "Folder-C": FOLDER_C.resolve(),
        "Folder-D": FOLDER_D.resolve(),
    }

    # No folder can be identical to, or nested inside, another.
    items = list(folders.items())
    for i, (name1, path1) in enumerate(items):
        for name2, path2 in items[i + 1:]:
            if path1 == path2 or path1 in path2.parents or path2 in path1.parents:
                raise ValueError(
                    f"{name1} and {name2} must be separate, non-nested folders:\n"
                    f"{path1}\n{path2}"
                )


# ==================================================
# 3. GPU SETUP AND INSIGHTFACE INITIALIZATION
# ==================================================

def initialize_insightface():
    global app

    print("\n========== GPU INFORMATION ==========")
    print("ONNX Runtime:", ort.__version__)

    # Load CUDA/cuDNN DLLs before creating ONNX sessions, when supported.
    try:
        ort.preload_dlls()
        print("CUDA/cuDNN DLL preload completed.")
    except (AttributeError, Exception) as exc:
        print("DLL preload skipped/information:", exc)

    available = ort.get_available_providers()
    print("Available providers:", available)

    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is not available. "
            "Check onnxruntime-gpu and NVIDIA CUDA/cuDNN dependencies."
        )

    print("\nLoading InsightFace buffalo_l model...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

    print("\n========== MODEL SESSION PROVIDERS ==========")
    cuda_sessions = 0

    for task, model in app.models.items():
        session = getattr(model, "session", None)
        if session is not None:
            session_providers = session.get_providers()
            print(f"{task}: {session_providers}")
            if "CUDAExecutionProvider" in session_providers:
                cuda_sessions += 1
        else:
            print(f"{task}: session provider details unavailable")

    if cuda_sessions == 0:
        print(
            "WARNING: CUDA is available in ONNX Runtime, but no model session "
            "reported CUDAExecutionProvider. Check GPU dependencies."
        )
    else:
        print(f"CUDA is active in {cuda_sessions} model session(s).")

    print("\nInsightFace initialized successfully.")


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm == 0:
        return None

    return vector / norm


def get_faces(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("OpenCV could not read this image.")

    return app.get(image)


def get_image_files(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def face_area(face):
    return max(0, face.bbox[2] - face.bbox[0]) * max(
        0, face.bbox[3] - face.bbox[1]
    )


# ==================================================
# 5. CREATE REFERENCE EMBEDDINGS
# ==================================================

def create_reference_embeddings():
    embeddings = []
    reference_images = get_image_files(FOLDER_A)

    if not reference_images:
        raise ValueError(f"No supported images found in {FOLDER_A}")

    print(f"\nProcessing {len(reference_images)} reference image(s) in Folder-A...")

    for image_path in tqdm(reference_images, desc="Reference photos"):
        try:
            faces = get_faces(image_path)

            if not faces:
                print(f"\nNo face detected in reference: {image_path.name}")
                continue

            # Assumption: the known person is the largest face in each reference image.
            largest_face = max(faces, key=face_area)
            embedding = normalize_embedding(largest_face.embedding)

            if embedding is not None:
                embeddings.append(embedding)

        except Exception as exc:
            print(f"\nReference error ({image_path.name}): {exc}")

    if not embeddings:
        raise ValueError(
            "No valid reference face embeddings were created. "
            "Check Folder-A images and face visibility."
        )

    print(f"Reference embeddings created: {len(embeddings)}")
    return np.stack(embeddings).astype(np.float32)


# ==================================================
# 6. SIMILARITY
# ==================================================

def calculate_similarity(reference_embeddings, candidate_embedding):
    # Both vectors are L2-normalized, so dot product equals cosine similarity.
    scores = np.dot(reference_embeddings, candidate_embedding)
    return float(np.max(scores))


# ==================================================
# 7. AVOID OVERWRITING FILES
# ==================================================

def unique_destination(destination):
    if not destination.exists():
        return destination

    counter = 1
    while True:
        candidate = destination.with_name(
            f"{destination.stem}_{counter}{destination.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


# ==================================================
# 8. PROCESS FOLDER-B
# ==================================================

def scan_folder_b(reference_embeddings):
    FOLDER_C.mkdir(parents=True, exist_ok=True)
    FOLDER_D.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(FOLDER_B)

    if not image_files:
        print("\nFolder-B contains no supported images.")
        return

    print(f"\nScanning {len(image_files)} candidate image(s)...")

    matched = 0
    unmatched = 0
    failed = 0
    report_rows = []

    for image_path in tqdm(image_files, desc="Candidate photos"):
        try:
            faces = get_faces(image_path)
            max_similarity = 0.0

            # Compare every detected face in the candidate image.
            for face in faces:
                candidate_embedding = normalize_embedding(face.embedding)
                if candidate_embedding is None:
                    continue

                score = calculate_similarity(
                    reference_embeddings, candidate_embedding
                )
                max_similarity = max(max_similarity, score)

            if max_similarity >= THRESHOLD:
                destination_folder = FOLDER_C
                status = "POTENTIAL_MATCH"
                matched += 1
            else:
                destination_folder = FOLDER_D
                status = "BELOW_THRESHOLD" if faces else "NO_FACE_DETECTED"
                unmatched += 1

            relative_path = image_path.relative_to(FOLDER_B)
            destination = unique_destination(destination_folder / relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if not DRY_RUN:
                shutil.move(str(image_path), str(destination))

            report_rows.append([
                str(image_path),
                str(destination),
                round(max_similarity, 4),
                status,
            ])

        except Exception as exc:
            failed += 1
            report_rows.append([
                str(image_path),
                "",
                "",
                f"FAILED: {exc}",
            ])
            print(f"\nFailed: {image_path.name}: {exc}")

    # Write the report even if some images failed.
    with REPORT_FILE.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Source", "Destination", "Similarity", "Status"])
        writer.writerows(report_rows)

    remaining = len(get_image_files(FOLDER_B))

    print("\n========== FINAL SUMMARY ==========")
    print("Total scanned:", len(image_files))
    print("Potential matches:", matched)
    print("Below threshold / no face:", unmatched)
    print("Failed:", failed)
    print("Supported images remaining in Folder-B:", remaining)
    print("CSV report:", REPORT_FILE)

    if DRY_RUN:
        print("\nDRY RUN: No photos were moved.")
        print("After reviewing the report, set DRY_RUN = False and run again.")
    elif remaining == 0:
        print("\nSUCCESS: No supported images remain in Folder-B.")
    else:
        print("\nSome images remain in Folder-B. Check FAILED rows in the CSV.")


# ==================================================
# 9. MAIN
# ==================================================

if __name__ == "__main__":
    print("\n======================================")
    print("         GPU FACE MATCHING SYSTEM")
    print("======================================")

    validate_folders()
    initialize_insightface()

    reference_embeddings = create_reference_embeddings()
    scan_folder_b(reference_embeddings)

    print("\nProcessing completed.")
