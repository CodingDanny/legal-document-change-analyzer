from pydantic import BaseModel, Field
from typing import Literal

class DecisionResult(BaseModel):
    justification: str = Field(..., min_length=1, max_length=400,
                               description="1–3 sentence rationale for category selection.")
    decision: Literal["Critical", "Minor", "Formatting"]

class ImpactAnalysis(BaseModel):
    legal_implications: str = Field(
        ..., min_length=1, max_length=600,
        description="Brief explanation of potential legal implications."
    )
    affected_party: Literal["Data Controller", "Processor", "Data Controller and Processor"]
    severity: Literal["high", "medium", "low"]


CHANGE_CLASSIFICATION_INSTRUCTIONS = """"You are a legal expert.

You will receive each contract modification in the following structure:
- Context before the change: <UNCHANGED TEXT IMMEDIATELY BEFORE THE CHANGE>
- Change:
   - Type: Modification | Deletion | Addition
   - Old: <TEXT IN THE OLD VERSION> (blank if not applicable)
   - New: <TEXT IN NEW VERSION> (blank if not applicable)
- Context after the change: <UNCHANGED TEXT IMMEDIATELY AFTER THE CHANGE>

Your task is to classify the modification into one of three categories:
- **Critical (Legally Significant):** Alters material rights, obligations, or legal meaning.
- **Minor (Non-substantive):** Clarifies intent, improves language, or updates administrative details without changing legal substance.
- **Formatting:** Involves only presentation aspects such as punctuation, capitalization, or layout.

Provide an evaluation in the following JSON format:
{
"justification": "1-3 sentence rationale for category selection.",
"decision": "Critical" | "Minor" | "Formatting"
}
- `justification`: Brief evaluation of the change w.r.t. to the categories, `decision`: The classification into Critical, Minor or Formatting.
- Do not easily give the highest category 'Critical', i.e. only categorize a change this way when it REALLY alters material rights, obligations, or legal meaning
- Provide only the required JSON structure; do not include any text outside the JSON.

Examples:
{
    "justification": "The change only adds a comma, making it a formatting change.",
    "decision": "Formatting"
}

{
    "justification": "The change clarifies the delivery definition by adding expected information, so it is a minor change.",
    "decision": "Minor"
}

{
    "justification": "The deadline has been updated, which is a major legal alteration and therefore a critical change.",
    "decision": "Critical"
}

Limit reasoning and responses to be brief and focused; adhere strictly to the requested output format."""

def change_analysis_prompt(context_before: str, change_type: str, old_content: str, new_content: str, context_after: str):
    return f"""Evaluate the following change:
- Context before the change: ... {context_before[-300:0]}
- Change:
   - Type: {change_type}
   - Old: {old_content}
   - New: {new_content}
- Context after the change: {context_after[0:300]} ..."""

IMPACT_ANALYSIS_INSTRUCTIONS = """"You are a legal expert.

You will receive critical contract modification in the following structure:
- Context before the change: <UNCHANGED TEXT IMMEDIATELY BEFORE THE CHANGE>
- Change:
   - Type: Modification | Deletion | Addition
   - Old: <TEXT IN THE OLD VERSION> (blank if not applicable)
   - New: <TEXT IN NEW VERSION> (blank if not applicable)
- Context after the change: <UNCHANGED TEXT IMMEDIATELY AFTER THE CHANGE>
- Justification for critical change: <REASON WHY THIS CHANGE IS CRITICAL>

Your task is to analyse the legal impact of the change by providing
- a brief explanation of potential legal implications, 
- which party (e.g. Data Controller / Processor) is most affected by the change
- and a severity rating (high / medium / low).

Provide an impact analysis in the following JSON format:
{
"legal_implications": "1-3 sentence describing potential legal implications.",
"affected_party": "Data Controller" | "Processor" | "Data Controller and Processor"
"severity": "high" | "medium" | "low"
}
- `legal_implications`: Brief explanation of legal implications, `affected_party`: The affected party (Data Controller / Processor / Data Controller and Processor), `severity`: The severity rating (high / medium / low) - don't be overly cautious, i.e. do not overestimate the severity and rather use medium as the default rating.
- Provide only the required JSON structure; do not include any text outside the JSON.

Examples:
{
    "legal_implications": "The clause weakens the processor’s mandatory security obligations by changing a 'shall' duty to a discretionary 'may'. Making this optional risks non-compliance and enforcement exposure for the controller.",
    "affected_party": "Data Controller",
    "severity": "high"
}

{
    "legal_implications": "This change relaxes a contractual 24-hour breach-notice SLA to the GDPR baseline of 'without undue delay'. Processors must still notify the controller without undue delay, but reduces assurance that the controller will receive notice quickly enough to meet its own 72-hour regulatory reporting obligation.",
    "affected_party": "Data Controller",
    "severity": "medium"
}

{
    "legal_implications": "This adds a non-exhaustive example that aligns with GDPR’s risk-based security duties (e.g., pseudonymisation/encryption). It does not change the underlying obligation to implement appropriate TOMs, and is consistent with the illustration of measures ‘such as encryption’.",
    "affected_party": "Processor",
    "severity": "low"
}

Limit reasoning and responses to be brief and focused; adhere strictly to the requested output format."""

def legal_implication_prompt(context_before: str, change_type: str, old_content: str, new_content: str, context_after: str, justification: str):
    return f"""Evaluate the following change:
- Context before the change: ... {context_before[-300:0]}
- Change:
   - Type: {change_type}
   - Old: {old_content}
   - New: {new_content}
- Context after the change: {context_after[0:300]} ...
- Justification for critical change: {justification}"""