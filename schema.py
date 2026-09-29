from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SoilHealthProfile(BaseModel):
    nitrogen_kg_ha: float = 140.0
    phosphorus_kg_ha: float = 22.5
    potassium_kg_ha: float = 190.0
    ph_level: float = 6.8
    organic_carbon_percentage: float = 0.75


class GeospatialIndices(BaseModel):
    ndvi: float = 0.68
    ndwi: float = 0.22
    canopy_water_content: float = 64.5


class WeatherContext(BaseModel):
    temperature_c: float = 31.5
    relative_humidity_pct: float = 68.0
    rainfall_forecast_mm_7d: float = 45.0
    agro_climatic_zone: str = "Plains & Coastal Agriculture Zone"


class UFSIDataPayload(BaseModel):
    version: str = "2026.1"
    state_code: str = "IN-GJ"
    farmer_id: str = "FID-8839201"
    plot_geometry_geojson: Dict[str, Any] = {"type": "Point", "coordinates": [72.5714, 23.0225]}
    soil_profile: SoilHealthProfile = SoilHealthProfile()
    remote_sensing_data: GeospatialIndices = GeospatialIndices()
    weather_context: WeatherContext = WeatherContext()


class RegenerativePractice(BaseModel):
    practice_name: str
    category: str
    impact_metrics: str
    implementation_steps: List[str]


class CropAdvisoryResponse(BaseModel):
    farmer_id: str
    farmer_name: str
    land_holding_acres: float
    plot_id: str
    recommended_crop: str
    crop_rotation_strategy: str
    regenerative_practices: List[RegenerativePractice]
    fertilizer_blend_recommendation: str
    water_conservation_strategy: str
    localized_advisory_text: str
    language_code: str


class DiseaseDiagnosticResult(BaseModel):
    is_plant_leaf: bool = Field(..., description="True if image is a crop or plant leaf, False if non-leaf/unrelated object.")
    plant_species: str = Field(default="Unknown", description="Species of crop identified in photo.")
    disease_identified: str = Field(..., description="Pathology name or Healthy.")
    confidence_score: float = Field(..., description="0.0 to 1.0 confidence score.")
    severity_level: str = Field(..., description="Low, Medium, High, Critical, or N/A.")
    symptoms: List[str] = Field(default_factory=list)
    organic_remediation: List[str] = Field(default_factory=list)
    chemical_remediation: List[str] = Field(default_factory=list)
    preventative_measures: List[str] = Field(default_factory=list)
    warning_message: Optional[str] = None