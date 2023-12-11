from pydantic import BaseModel

class Device_Input(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Conductivity: float
    Turbidity: float 