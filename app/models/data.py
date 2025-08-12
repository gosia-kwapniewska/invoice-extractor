from pydantic import BaseModel

class InvoiceData(BaseModel):
    consignor: str | None = None
    consignee: str | None = None
    country_of_origin: str | None = None
    country_of_destination: str | None = None
    hs_code: str | None = None
    description_of_goods: str | None = None
    means_of_transport: str | None = None
    vessel: str | None = None