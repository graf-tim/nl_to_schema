from pydantic import BaseModel, Field


class Entitaet(BaseModel):
    name: str
    attribute: list[str]
    beschreibung: str


class Beziehung(BaseModel):
    von: str
    zu: str
    kardinalitaet: str
    beschreibung: str


class Unklarheit(BaseModel):
    beschreibung: str
    betroffene_elemente: list[str]


class RequirementsReport(BaseModel):
    entitaeten: list[Entitaet]
    beziehungen: list[Beziehung]
    unklarheiten: list[Unklarheit] = Field(default_factory=list)


class ERAttribut(BaseModel):
    name: str
    datentyp: str
    primaerschluessel: bool = False
    pflichtfeld: bool = True


class EREntitaet(BaseModel):
    name: str
    attribute: list[ERAttribut]


class ERBeziehung(BaseModel):
    von: str
    zu: str
    kardinalitaet: str
    beziehungsattribute: list[str] = Field(default_factory=list)


class ERModell(BaseModel):
    entitaeten: list[EREntitaet]
    beziehungen: list[ERBeziehung]
