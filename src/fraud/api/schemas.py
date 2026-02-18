from pydantic import BaseModel


class FraudInput(BaseModel):
    Income: float
    Age: int
    Experience: int
    Married_Single: str
    House_Ownership: str
    Car_Ownership: str
    Profession: str
    CITY: str
    STATE: str
    CURRENT_JOB_YRS: int
    CURRENT_HOUSE_YRS: int
