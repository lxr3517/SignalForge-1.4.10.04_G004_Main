from pydantic import BaseModel

class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    frequency: str = 'D'
    years_of_history: int = 5
    grouped_forecast: bool = False
    forecast_scope: str = 'company_total'
