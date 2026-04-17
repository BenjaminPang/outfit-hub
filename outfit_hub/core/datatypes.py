from typing import Optional, TypedDict, Union
from PIL import Image
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class FashionItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # item_id: Optional[int] = Field(
    #     default=None,
    #     description="Unique ID of the item, mapped to `id` in the ItemLoader"
    # )
    item_idx: Optional[int] = Field(
        default=None,
        description="Unique sequential index used for image retrieval and parquet data mapping."
    )
    category: Optional[str] = Field(
        default="",
        description="Category of the item"
    )
    image: Optional[Image.Image] = Field(
        default=None,
        description="Image of the item"
    )
    description: Optional[str] = Field(
        default="",
        description="Description of the item"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata for the item"
    )
    embedding: Optional[np.ndarray] = Field(
        default=None,
        description="Embedding of the item"
    )


class FashionWeightedItem(BaseModel):
    item: list[FashionItem]
    weight: list[float]


class FashionOutfit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    outfit: list[FashionItem] = Field(
        default_factory=list,
        description="List of fashion items"
    )


class FashionComplementaryQuery(BaseModel):
    outfit: list[FashionItem] = Field(
        default_factory=list,
        description="List of fashion items"
    )
    # category: str = Field(
    #     default="",
    #     description="Category of the target outfit"
    # )
    

class FashionCompatibilityData(TypedDict):
    label: Union[int, list[int]]
    query: Union[FashionOutfit, list[FashionOutfit]]
    
    
class FashionFillInTheBlankData(TypedDict):
    query: Union[FashionComplementaryQuery, list[FashionComplementaryQuery]]
    label: Union[int, list[int]]
    candidates: Union[list[FashionItem], list[list[FashionItem]]]


class FashionContrastivetData(TypedDict):
    query: Union[
        FashionComplementaryQuery,
        list[FashionComplementaryQuery]
    ]
    answer: Union[
        FashionItem,
        list[FashionItem],
        FashionWeightedItem,
        list[FashionWeightedItem]
    ]