from fastapi import APIRouter, Depends
from activetigger.tasks.compute_similarity import ComputeSimilarity

router = APIRouter()

@router.post("/elements/similar")
async def get_similar_elements(project_slug: str,element_id: str,k: int = 10):
    similarity = ComputeSimilarity(embeddings)
    results = similarity.search(query_vector, k=k)
    return results