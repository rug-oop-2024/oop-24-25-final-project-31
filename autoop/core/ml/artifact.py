from typing import Any, Dict, List
from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    asset_path: str = Field(..., description="The storage path of the asset")
    version: str = Field(..., description="Version of the artifact")
    data: Any = Field(..., description="Binary or serialized data")
    metadata: Dict[str, Any] = Field({}, description="Experiment and run ID")
    type: str = Field(..., description="Model of the artifact")
    tags: List[str] = Field([], description="Tags describing the artifact")
    id: str = Field(None, description="ID made from asset_path and version")

    def generate_id(cls, values):
        asset_path = values.get("asset_path")
        version = values.get("version")
        if asset_path and version:
            path = base64.urlsafe_b64encode(asset_path.encode()).decode()
            return f"{path}:{version}"
        raise ValueError("Asset_path and version are not provided")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "asset_path": self.asset_path,
            "version": self.version,
            "data": self.data,
            **self.metadata,
            "type": self.type,
            "tags": self.tags,
            "id": self.id
        }
