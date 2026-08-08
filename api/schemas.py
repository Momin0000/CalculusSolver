"""
Pydantic v2 schemas for CalculusSolver API payloads and responses.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class TermSchema(BaseModel):
    coeff: Union[int, float] = 1
    var: Optional[Dict[str, Union[int, float]]] = None
    func: Optional[str] = None
    arg: Optional[str] = None


class PolynomialSchema(BaseModel):
    terms: List[TermSchema] = Field(default_factory=list)


class FractionSchema(BaseModel):
    numi: Union[PolynomialSchema, TermSchema, Dict[str, Any], int, float]
    deno: Union[PolynomialSchema, TermSchema, Dict[str, Any], int, float] = 1


class SolveRequest(BaseModel):
    op: str = Field(..., description="Operation: diff, partial, integrate, gradient, hessian, tangent_line")
    var: str = Field(default="x", description="Variable of differentiation or integration")
    expr: Dict[str, Any] = Field(..., description="SLaNg fraction expression object")
    point: Optional[Dict[str, float]] = Field(default=None, description="Evaluation point for tangent_line or evaluation")


class ValidateRequest(BaseModel):
    expression: Dict[str, Any] = Field(..., description="SLaNg AST node or structure to validate")


class StepSchema(BaseModel):
    rule: str
    description: str
    before: str
    after: str


class SolveResponse(BaseModel):
    status: str = "solved"
    expr: Dict[str, Any]
    steps: List[StepSchema] = Field(default_factory=list)
    latex: str
    confidence: float = 1.0
    verified: bool = True
    warning: Optional[str] = None
    rule: Optional[str] = None
    mode: str = "fallback"


class HealthResponse(BaseModel):
    status: str = "ok"
    solver_mode: str = "fallback"
    solver_loaded: bool = True
    checkpoint_error: Optional[str] = None
