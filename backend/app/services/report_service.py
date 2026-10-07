import csv
import io
import json
from app.schemas.pipeline import PipelineResponseSchema

class ReportService:
    """Exports pipeline analysis results to tabular or document formats."""

    @staticmethod
    def export_to_json(result: PipelineResponseSchema) -> str:
        return result.model_dump_json(indent=2)

    @staticmethod
    def export_segments_to_csv(result: PipelineResponseSchema) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Segment ID", "Start Time (s)", "End Time (s)", "Language", "Transcript", "Sentiment", "Confidence"])

        sentiment_map = {s.segment_id: s for s in result.segment_sentiments}

        for seg in result.segments:
            sent_info = sentiment_map.get(seg.id)
            sent_label = sent_info.sentiment.value if sent_info else "unknown"
            sent_conf = sent_info.confidence if sent_info else 0.0

            writer.writerow([
                seg.id,
                seg.start_time,
                seg.end_time,
                seg.language_code,
                seg.text,
                sent_label,
                sent_conf
            ])

        return output.getvalue()

    @staticmethod
    def export_entities_to_csv(result: PipelineResponseSchema) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Entity ID", "Text", "Category", "Segment ID", "Start Time", "Confidence"])

        for ent in result.entities:
            writer.writerow([
                ent.id,
                ent.text,
                ent.category.value,
                ent.segment_id or "",
                ent.start_time or "",
                ent.confidence
            ])

        return output.getvalue()
