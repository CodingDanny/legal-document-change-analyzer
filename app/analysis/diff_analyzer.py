from typing import List, Any, Tuple, Dict, Optional
import copy
from app.diff.pdf_diff import Change
from app.analysis.analysis_prompt_util import (
    change_analysis_prompt, 
    DecisionResult, 
    CHANGE_CLASSIFICATION_INSTRUCTIONS,
    ImpactAnalysis,
    IMPACT_ANALYSIS_INSTRUCTIONS,
    legal_implication_prompt
)
from app.ai.openai_util import parallel_openai_calls


class DiffAnalyzer:
    def __init__(self, changes: List[Change]):
        self.changes = changes
        self.analyzed_changes = None
    
    def analyze_changes(self) -> None:
        """Analyze changes with AI and store results in analyzed_changes"""
        prompts_and_changes = self.generate_prompts_with_changes()
        
        if not prompts_and_changes:
            self.analyzed_changes = copy.deepcopy(self.changes)
            return
        
        prompts = [prompt for prompt, _ in prompts_and_changes]
        changes_to_analyze = [change for _, change in prompts_and_changes]
        
        try:
            ai_results = parallel_openai_calls(
                instructions=CHANGE_CLASSIFICATION_INSTRUCTIONS,
                prompts=prompts,
                text_format_class=DecisionResult
            )
        except Exception as e:
            print(f"Error during AI classification: {e}")
            ai_results = [None] * len(prompts)
        
        self.analyzed_changes = copy.deepcopy(self.changes)
        
        change_id_to_classification = {}
        for change, ai_result in zip(changes_to_analyze, ai_results):
            if ai_result:
                change_id_to_classification[change.change_id] = {
                    "decision": ai_result.decision,
                    "justification": ai_result.justification
                }
            else:
                change_id_to_classification[change.change_id] = {
                    "decision": "ERROR",
                    "justification": "Failed to classify"
                }
        
        for change in self.analyzed_changes:
            if change.change_id in change_id_to_classification:
                change.classification = change_id_to_classification[change.change_id]
        
        self.analyze_critical_impacts()
    
    def analyze_critical_impacts(self) -> None:
        """Analyze legal impacts for critical changes"""
        if self.analyzed_changes is None:
            return
        
        prompts_and_changes = self.generate_impact_prompts_with_changes()
        
        if not prompts_and_changes:
            return
        
        prompts = [prompt for prompt, _ in prompts_and_changes]
        critical_changes = [change for _, change in prompts_and_changes]
        
        try:
            ai_results = parallel_openai_calls(
                instructions=IMPACT_ANALYSIS_INSTRUCTIONS,
                prompts=prompts,
                text_format_class=ImpactAnalysis
            )
        except Exception as e:
            print(f"Error during impact analysis: {e}")
            ai_results = [None] * len(prompts)
        
        change_id_to_impact = {}
        for change, ai_result in zip(critical_changes, ai_results):
            if ai_result:
                change_id_to_impact[change.change_id] = {
                    "legal_implications": ai_result.legal_implications,
                    "affected_party": ai_result.affected_party,
                    "severity": ai_result.severity
                }
        
        for change in self.analyzed_changes:
            if change.change_id in change_id_to_impact:
                change.impact_analysis = change_id_to_impact[change.change_id]
    
    def generate_impact_prompts_with_changes(self) -> List[Tuple[str, Change]]:
        """Generate prompts for critical changes that need impact analysis"""
        if self.analyzed_changes is None:
            return []
        
        critical_changes = []
        critical_indices = []
        
        for i, change in enumerate(self.analyzed_changes):
            if (hasattr(change, 'classification') and 
                change.classification.get('decision') == 'Critical'):
                critical_changes.append(change)
                critical_indices.append(i)
        
        prompts_and_changes = []
        for critical_change, original_index in zip(critical_changes, critical_indices):
            context_before = self._get_context_before(original_index)
            context_after = self._get_context_after(original_index)
            justification = critical_change.classification.get('justification', '')
            prompt = self._generate_impact_prompt_for_change(
                critical_change, context_before, context_after, justification
            )
            prompts_and_changes.append((prompt, critical_change))
        
        return prompts_and_changes
    
    def _generate_impact_prompt_for_change(self, change: Change, context_before: str, 
                                          context_after: str, justification: str) -> str:
        """Generate impact analysis prompt for a critical change"""
        change_type = ""
        old_content = ""
        new_content = ""
        
        if change.type == "modified":
            change_type = "Modification"
            old_content = self._merge_content_list(change.old_content)
            new_content = self._merge_content_list(change.new_content)
        
        elif change.type == "moved_and_modified":
            change_type = "Modification"
            old_content = change.old_content if change.old_content else ""
            new_content = change.new_content if change.new_content else ""
        
        elif change.type == "added":
            change_type = "Addition"
            old_content = ""
            new_content = self._merge_content_list(change.content)
        
        elif change.type == "removed":
            change_type = "Deletion"
            old_content = self._merge_content_list(change.content)
            new_content = ""
        
        return legal_implication_prompt(
            context_before=context_before,
            change_type=change_type,
            old_content=old_content,
            new_content=new_content,
            context_after=context_after,
            justification=justification
        )
    
    def generate_prompts(self) -> List[str]:
        prompts_and_changes = self.generate_prompts_with_changes()
        return [prompt for prompt, _ in prompts_and_changes]
    
    def generate_prompts_with_changes(self) -> List[Tuple[str, Change]]:
        filtered_changes = []
        filtered_indices = []
        
        for i, change in enumerate(self.changes):
            if change.change_id is not None:
                filtered_changes.append(change)
                filtered_indices.append(i)
        
        prompts_and_changes = []
        for filtered_change, original_index in zip(filtered_changes, filtered_indices):
            context_before = self._get_context_before(original_index)
            context_after = self._get_context_after(original_index)
            prompt = self._generate_prompt_for_change(filtered_change, context_before, context_after)
            prompts_and_changes.append((prompt, filtered_change))
        
        return prompts_and_changes
    
    def _get_context_before(self, index: int) -> str:
        if index > 0:
            predecessor = self.changes[index - 1]
            if predecessor.type == "unchanged":
                return self._merge_content_list(predecessor.content)
        return ""
    
    def _get_context_after(self, index: int) -> str:
        if index < len(self.changes) - 1:
            successor = self.changes[index + 1]
            if successor.type == "unchanged":
                return self._merge_content_list(successor.content)
        return ""
    
    def _merge_content_list(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)
    
    def _generate_prompt_for_change(self, change: Change, context_before: str, context_after: str) -> str:
        change_type = ""
        old_content = ""
        new_content = ""
        
        if change.type == "modified":
            change_type = "Modification"
            old_content = self._merge_content_list(change.old_content)
            new_content = self._merge_content_list(change.new_content)
        
        elif change.type == "moved_and_modified":
            change_type = "Modification"
            old_content = change.old_content if change.old_content else ""
            new_content = change.new_content if change.new_content else ""
        
        elif change.type == "added":
            change_type = "Addition"
            old_content = ""
            new_content = self._merge_content_list(change.content)
        
        elif change.type == "removed":
            change_type = "Deletion"
            old_content = self._merge_content_list(change.content)
            new_content = ""
        
        return change_analysis_prompt(
            context_before=context_before,
            change_type=change_type,
            old_content=old_content,
            new_content=new_content,
            context_after=context_after
        )
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics of classifications"""
        if self.analyzed_changes is None:
            return {}
        
        summary = {
            "Critical": 0,
            "Minor": 0,
            "Formatting": 0,
            "ERROR": 0,
            "total_analyzed": 0
        }
        
        for change in self.analyzed_changes:
            if hasattr(change, 'classification'):
                decision = change.classification.get("decision", "ERROR")
                if decision in summary:
                    summary[decision] += 1
                summary["total_analyzed"] += 1
        
        return summary
    
    def to_dict(self) -> List[Dict]:
        """Export analyzed changes as list of dictionaries"""
        if self.analyzed_changes is None:
            return []
        
        result = []
        for change in self.analyzed_changes:
            change_dict = change.to_dict()
            if hasattr(change, 'classification'):
                change_dict['classification'] = change.classification
            if hasattr(change, 'impact_analysis'):
                change_dict['impact_analysis'] = change.impact_analysis
            result.append(change_dict)
        
        return result