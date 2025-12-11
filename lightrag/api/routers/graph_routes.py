"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional, Dict, Any
import traceback
from fastapi import APIRouter, Depends, Query, HTTPException, Request
from pydantic import BaseModel

from lightrag.utils import logger
from ..utils_api import get_combined_auth_dependency
from ..dependencies import get_current_workspace

router = APIRouter(tags=["graph"])


async def get_rag_for_request(request: Request, rag_instance=None):
    """
    Get the appropriate LightRAG instance for the request.

    In single-instance mode, returns the passed rag_instance.
    In multi-tenant mode, resolves the workspace and gets the appropriate instance.

    Note: Graph operations are performed on LightRAG instances, so we use
    get_lightrag_instance() to ensure we always get a LightRAG instance.
    """
    workspace_manager = getattr(request.app.state, "workspace_manager", None)

    if workspace_manager is not None:
        # Multi-tenant mode - get workspace-specific LightRAG instance
        workspace = await get_current_workspace(request)
        return await workspace_manager.get_lightrag_instance(workspace)
    else:
        # Single-instance mode - use the provided rag instance
        return rag_instance


class EntityUpdateRequest(BaseModel):
    entity_name: str
    updated_data: Dict[str, Any]
    allow_rename: bool = False


class RelationUpdateRequest(BaseModel):
    source_id: str
    target_id: str
    updated_data: Dict[str, Any]


def create_graph_routes(rag, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    # Store rag instance for closure access in multi-tenant helper
    _default_rag = rag

    @router.get("/graph/label/list", dependencies=[Depends(combined_auth)])
    async def get_graph_labels(http_request: Request):
        """
        Get all graph labels

        Returns:
            List[str]: List of graph labels
        """
        try:
            rag_instance = await get_rag_for_request(http_request, _default_rag)
            return await rag_instance.get_graph_labels()
        except Exception as e:
            logger.error(f"Error getting graph labels: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error getting graph labels: {str(e)}"
            )

    @router.get("/graphs", dependencies=[Depends(combined_auth)])
    async def get_knowledge_graph(
        http_request: Request,
        label: str = Query(..., description="Label to get knowledge graph for"),
        max_depth: int = Query(3, description="Maximum depth of graph", ge=1),
        max_nodes: int = Query(1000, description="Maximum nodes to return", ge=1),
    ):
        """
        Retrieve a connected subgraph of nodes where the label includes the specified label.
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. Hops(path) to the staring node take precedence
            2. Followed by the degree of the nodes

        Args:
            label (str): Label of the starting node
            max_depth (int, optional): Maximum depth of the subgraph,Defaults to 3
            max_nodes: Maxiumu nodes to return

        Returns:
            Dict[str, List[str]]: Knowledge graph for label
        """
        try:
            rag_instance = await get_rag_for_request(http_request, _default_rag)
            # Log the label parameter to check for leading spaces
            logger.debug(
                f"get_knowledge_graph called with label: '{label}' (length: {len(label)}, repr: {repr(label)})"
            )

            return await rag_instance.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.error(f"Error getting knowledge graph for label '{label}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error getting knowledge graph: {str(e)}"
            )

    @router.get("/graph/entity/exists", dependencies=[Depends(combined_auth)])
    async def check_entity_exists(
        http_request: Request,
        name: str = Query(..., description="Entity name to check"),
    ):
        """
        Check if an entity with the given name exists in the knowledge graph

        Args:
            name (str): Name of the entity to check

        Returns:
            Dict[str, bool]: Dictionary with 'exists' key indicating if entity exists
        """
        try:
            rag_instance = await get_rag_for_request(http_request, _default_rag)
            exists = await rag_instance.chunk_entity_relation_graph.has_node(name)
            return {"exists": exists}
        except Exception as e:
            logger.error(f"Error checking entity existence for '{name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error checking entity existence: {str(e)}"
            )

    @router.post("/graph/entity/edit", dependencies=[Depends(combined_auth)])
    async def update_entity(request: EntityUpdateRequest, http_request: Request):
        """
        Update an entity's properties in the knowledge graph

        Args:
            request (EntityUpdateRequest): Request containing entity name, updated data, and rename flag

        Returns:
            Dict: Updated entity information
        """
        try:
            rag_instance = await get_rag_for_request(http_request, _default_rag)
            result = await rag_instance.aedit_entity(
                entity_name=request.entity_name,
                updated_data=request.updated_data,
                allow_rename=request.allow_rename,
            )
            return {
                "status": "success",
                "message": "Entity updated successfully",
                "data": result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error updating entity '{request.entity_name}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error updating entity '{request.entity_name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error updating entity: {str(e)}"
            )

    @router.post("/graph/relation/edit", dependencies=[Depends(combined_auth)])
    async def update_relation(request: RelationUpdateRequest, http_request: Request):
        """Update a relation's properties in the knowledge graph

        Args:
            request (RelationUpdateRequest): Request containing source ID, target ID and updated data

        Returns:
            Dict: Updated relation information
        """
        try:
            rag_instance = await get_rag_for_request(http_request, _default_rag)
            result = await rag_instance.aedit_relation(
                source_entity=request.source_id,
                target_entity=request.target_id,
                updated_data=request.updated_data,
            )
            return {
                "status": "success",
                "message": "Relation updated successfully",
                "data": result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error updating relation between '{request.source_id}' and '{request.target_id}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(
                f"Error updating relation between '{request.source_id}' and '{request.target_id}': {str(e)}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error updating relation: {str(e)}"
            )

    return router
