from typing import List, Optional
from pydantic import BaseModel

class ModelBreakdown(BaseModel):
    df40_clip_score: Optional[float] = None
    vit_score: Optional[float] = None
    efficientnet_score: Optional[float] = None

class ImageDetectionResponse(BaseModel):
    verdict: str; confidence: float; fake_probability: float; risk_level: str
    model_used: Optional[str] = None
    model_breakdown: ModelBreakdown
    heatmap_overlay: Optional[str] = None
    face_detected: bool; processing_time_sec: float

class FrameStats(BaseModel):
    frames_sampled: int; frames_with_face_detected: int
    mean_fake_probability: float; max_fake_probability: float; std_fake_probability: float

class MostSuspiciousFrame(BaseModel):
    frame_index: int; fake_probability: float; heatmap_overlay: Optional[str] = None

class BlinkAnalysis(BaseModel):
    signal: str; note: str

class PerFrameResult(BaseModel):
    frame_idx: int; fake_probability: float; verdict: str

class VideoDetectionResponse(BaseModel):
    verdict: str; confidence: float; fake_probability: float; risk_level: str
    frame_stats: FrameStats; most_suspicious_frame: MostSuspiciousFrame
    blink_analysis: BlinkAnalysis; per_frame_results: List[PerFrameResult]
    processing_time_sec: float

class ErrorResponse(BaseModel):
    detail: str