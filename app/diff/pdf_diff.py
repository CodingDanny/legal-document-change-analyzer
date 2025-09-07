import pymupdf4llm
import pymupdf
import patiencediff
from diff_match_patch import diff_match_patch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class Change:
    type: str
    old_range: Optional[List[int]] = None
    new_range: Optional[List[int]] = None
    old_index: Optional[int] = None
    new_index: Optional[int] = None
    content: Optional[Any] = None
    old_content: Optional[Any] = None
    new_content: Optional[Any] = None
    similarity: Optional[float] = None
    inline_diff: Optional[List[Tuple[int, str]]] = None
    change_id: Optional[int] = None
    
    def to_dict(self) -> Dict:
        result = {'type': self.type}
        for key, value in self.__dict__.items():
            if key != 'type' and value is not None:
                result[key] = value
        return result


@dataclass
class Statistics:
    total_blocks_old: int
    total_blocks_new: int
    unchanged: int = 0
    modified: int = 0
    added: int = 0
    removed: int = 0
    moved: int = 0
    moved_and_modified: int = 0


@dataclass
class Metadata:
    generated_at: str
    diff_version: str
    similarity_threshold: float


@dataclass
class DiffResult:
    changes: List[Change]
    statistics: Statistics
    metadata: Metadata


class BlockExtractor:
    def extract_blocks_from_pdf(self, pdf_data: bytes) -> List[str]:
        doc = pymupdf.Document(stream=pdf_data, filetype='pdf')
        markdown = pymupdf4llm.to_markdown(doc)
        doc.close()
        return self.split_markdown_into_blocks(markdown)
    
    def split_markdown_into_blocks(self, markdown: str) -> List[str]:
        blocks = []
        current_paragraph_lines = []
        
        for line in markdown.split('\n'):
            if self.is_header_line(line):
                if current_paragraph_lines:
                    blocks.append('\n'.join(current_paragraph_lines))
                    current_paragraph_lines = []
                blocks.append(line)
            elif self.is_paragraph_separator(line):
                if current_paragraph_lines:
                    blocks.append('\n'.join(current_paragraph_lines))
                    current_paragraph_lines = []
            else:
                current_paragraph_lines.append(line)
        
        if current_paragraph_lines:
            blocks.append('\n'.join(current_paragraph_lines))
        
        return [block for block in blocks if block.strip()]
    
    def is_header_line(self, line: str) -> bool:
        return line.startswith('#')
    
    def is_paragraph_separator(self, line: str) -> bool:
        return line.strip() == ''


class SimilarityCalculator:
    def __init__(self):
        self.dmp = diff_match_patch()
        self.dmp.Diff_Timeout = 0
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        
        diffs = self.dmp.diff_main(text1, text2)
        self.dmp.diff_cleanupSemantic(diffs)
        
        max_length = max(len(text1), len(text2))
        if max_length == 0:
            return 1.0
        
        unchanged_length = sum(len(text) for op, text in diffs if op == 0)
        return unchanged_length / max_length
    
    def generate_inline_diff(self, old_text: str, new_text: str) -> List[Tuple[int, str]]:
        diffs = self.dmp.diff_main(old_text, new_text)
        self.dmp.diff_cleanupSemantic(diffs)
        return [(op, text) for op, text in diffs]


class MoveDetector:
    def __init__(self, similarity_threshold: float, min_words_for_move: int):
        self.similarity_threshold = similarity_threshold
        self.min_words_for_move = min_words_for_move
        self.similarity_calc = SimilarityCalculator()
    
    def find_potential_moves(self, opcodes: List[Tuple], blocks1: List[str], blocks2: List[str]) -> Dict:
        removed_candidates = self.collect_removed_candidates(opcodes, blocks1)
        added_candidates = self.collect_added_candidates(opcodes, blocks2)
        return self.match_candidates(removed_candidates, added_candidates)
    
    def collect_removed_candidates(self, opcodes: List[Tuple], blocks: List[str]) -> List[Tuple[int, str]]:
        candidates = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'delete':
                for i in range(i1, i2):
                    if self.is_eligible_for_move(blocks[i]):
                        candidates.append((i, blocks[i]))
        return candidates
    
    def collect_added_candidates(self, opcodes: List[Tuple], blocks: List[str]) -> List[Tuple[int, str]]:
        candidates = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                for j in range(j1, j2):
                    if self.is_eligible_for_move(blocks[j]):
                        candidates.append((j, blocks[j]))
        return candidates
    
    def is_eligible_for_move(self, block_text: str) -> bool:
        word_count = len(block_text.split())
        return word_count >= self.min_words_for_move
    
    def match_candidates(self, removed: List[Tuple[int, str]], added: List[Tuple[int, str]]) -> Dict:
        moves = {}
        for removed_idx, removed_text in removed:
            matches = self.find_matches_for_block(removed_text, added)
            if matches:
                moves[(removed_idx, removed_text)] = matches
        return moves
    
    def find_matches_for_block(self, removed_text: str, added_candidates: List[Tuple[int, str]]) -> List[Tuple[int, str, float]]:
        matches = []
        for added_idx, added_text in added_candidates:
            similarity = self.similarity_calc.calculate_similarity(removed_text, added_text)
            if similarity >= self.similarity_threshold:
                matches.append((added_idx, added_text, similarity))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def create_move_change(self, old_idx: int, new_idx: int, content: str) -> Change:
        return Change(
            type='moved',
            old_index=old_idx,
            new_index=new_idx,
            content=content
        )
    
    def create_moved_and_modified_change(self, old_idx: int, new_idx: int, 
                                        old_text: str, new_text: str, 
                                        similarity: float) -> Change:
        inline_diff = self.similarity_calc.generate_inline_diff(old_text, new_text)
        return Change(
            type='moved_and_modified',
            old_index=old_idx,
            new_index=new_idx,
            old_content=old_text,
            new_content=new_text,
            similarity=similarity,
            inline_diff=inline_diff
        )


class ChangeDetector:
    def __init__(self, blocks1: List[str], blocks2: List[str], 
                 similarity_threshold: float, min_words_for_move: int):
        self.blocks1 = blocks1
        self.blocks2 = blocks2
        self.move_detector = MoveDetector(similarity_threshold, min_words_for_move)
        self.similarity_calc = SimilarityCalculator()
    
    def detect_changes(self) -> List[Change]:
        opcodes = self.generate_opcodes()
        potential_moves = self.move_detector.find_potential_moves(opcodes, self.blocks1, self.blocks2)
        
        changes = []
        used_sources = set()
        used_targets = set()
        
        for opcode in opcodes:
            tag, i1, i2, j1, j2 = opcode
            
            if tag == 'equal':
                change = self.process_equal_blocks(i1, i2, j1, j2)
                changes.append(change)
            
            elif tag == 'replace':
                change = self.process_replacement(i1, i2, j1, j2)
                changes.append(change)
            
            elif tag == 'delete':
                deletion_changes = self.process_deletions(
                    i1, i2, j1, potential_moves, used_sources, used_targets
                )
                changes.extend(deletion_changes)
            
            elif tag == 'insert':
                insertion_changes = self.process_insertions(j1, j2, used_targets)
                changes.extend(insertion_changes)
        
        return self.remove_duplicate_additions(changes)
    
    def generate_opcodes(self) -> List[Tuple]:
        matcher = patiencediff.PatienceSequenceMatcher(None, self.blocks1, self.blocks2)
        return list(matcher.get_opcodes())
    
    def process_equal_blocks(self, i1: int, i2: int, j1: int, j2: int) -> Change:
        return Change(
            type='unchanged',
            old_range=[i1, i2],
            new_range=[j1, j2],
            content=self.blocks1[i1:i2]
        )
    
    def process_replacement(self, i1: int, i2: int, j1: int, j2: int) -> Change:
        old_text = '\n'.join(self.blocks1[i1:i2])
        new_text = '\n'.join(self.blocks2[j1:j2])
        
        similarity = self.similarity_calc.calculate_similarity(old_text, new_text)
        inline_diff = self.similarity_calc.generate_inline_diff(old_text, new_text)
        
        return Change(
            type='modified',
            old_range=[i1, i2],
            new_range=[j1, j2],
            old_content=self.blocks1[i1:i2],
            new_content=self.blocks2[j1:j2],
            similarity=similarity,
            inline_diff=inline_diff
        )
    
    def process_deletions(self, i1: int, i2: int, j1: int, 
                         potential_moves: Dict, used_sources: set, 
                         used_targets: set) -> List[Change]:
        changes = []
        
        for i in range(i1, i2):
            move_change = self.try_detect_move(
                i, potential_moves, used_sources, used_targets
            )
            
            if move_change:
                changes.append(move_change)
            else:
                removed_change = self.create_removal(i, j1)
                changes.append(removed_change)
        
        return changes
    
    def try_detect_move(self, block_idx: int, potential_moves: Dict, 
                        used_sources: set, used_targets: set) -> Optional[Change]:
        block_text = self.blocks1[block_idx]
        
        for (src_idx, src_text), matches in potential_moves.items():
            if src_idx == block_idx and block_idx not in used_sources:
                for tgt_idx, tgt_text, similarity in matches:
                    if tgt_idx not in used_targets:
                        used_sources.add(block_idx)
                        used_targets.add(tgt_idx)
                        
                        if similarity == 1.0:
                            return self.move_detector.create_move_change(
                                block_idx, tgt_idx, block_text
                            )
                        else:
                            return self.move_detector.create_moved_and_modified_change(
                                block_idx, tgt_idx, src_text, tgt_text, similarity
                            )
        return None
    
    def create_removal(self, old_idx: int, new_position_hint: int) -> Change:
        change = Change(
            type='removed',
            old_index=old_idx,
            content=self.blocks1[old_idx]
        )
        change._new_position_hint = new_position_hint
        return change
    
    def process_insertions(self, j1: int, j2: int, used_targets: set) -> List[Change]:
        changes = []
        for j in range(j1, j2):
            if j not in used_targets:
                change = Change(
                    type='added',
                    new_index=j,
                    content=self.blocks2[j]
                )
                changes.append(change)
        return changes
    
    def remove_duplicate_additions(self, changes: List[Change]) -> List[Change]:
        moved_indices = set()
        for change in changes:
            if change.type == 'moved' and change.new_index is not None:
                moved_indices.add(change.new_index)
        
        filtered_changes = []
        for change in changes:
            if change.type == 'added' and change.new_index in moved_indices:
                continue
            filtered_changes.append(change)
        
        return filtered_changes


class ChangeOrganizer:
    def organize_changes(self, changes: List[Change]) -> List[Change]:
        sorted_changes = self.sort_by_document_order(changes)
        consolidated = self.consolidate_consecutive_changes(sorted_changes)
        numbered_changes = self.assign_change_ids(consolidated)
        return numbered_changes
    
    def assign_change_ids(self, changes: List[Change]) -> List[Change]:
        change_id_counter = 1
        for change in changes:
            if change.type != 'unchanged' and change.type != 'moved':
                change.change_id = change_id_counter
                change_id_counter += 1
        return changes
    
    def sort_by_document_order(self, changes: List[Change]) -> List[Change]:
        def get_sort_position(change: Change) -> Tuple[float, int]:
            if change.type == 'removed' and hasattr(change, '_new_position_hint'):
                return (change._new_position_hint - 0.1, 0)
            
            if change.new_range:
                return (change.new_range[0], 0)
            elif change.new_index is not None:
                return (change.new_index, 0)
            
            if change.old_range:
                return (999999 + change.old_range[0], 0)
            elif change.old_index is not None:
                return (999999 + change.old_index, 0)
            
            return (999999, 0)
        
        sorted_changes = sorted(changes, key=get_sort_position)
        
        for change in sorted_changes:
            if hasattr(change, '_new_position_hint'):
                delattr(change, '_new_position_hint')
        
        return sorted_changes
    
    def consolidate_consecutive_changes(self, changes: List[Change]) -> List[Change]:
        if not changes:
            return []
        
        consolidated = []
        i = 0
        
        while i < len(changes):
            current = changes[i]
            
            if current.type in ['removed', 'added', 'unchanged']:
                merged_change = self.try_merge_consecutive(changes, i)
                consolidated.append(merged_change)
                i = merged_change._merge_count if hasattr(merged_change, '_merge_count') else i + 1
                if hasattr(merged_change, '_merge_count'):
                    delattr(merged_change, '_merge_count')
            else:
                consolidated.append(current)
                i += 1
        
        return consolidated
    
    def try_merge_consecutive(self, changes: List[Change], start_idx: int) -> Change:
        current = changes[start_idx]
        j = start_idx + 1
        
        while j < len(changes) and changes[j].type == current.type:
            next_change = changes[j]
            
            if self.can_merge(current, next_change):
                self.merge_changes(current, next_change)
                j += 1
            else:
                break
        
        current._merge_count = j
        return current
    
    def can_merge(self, change1: Change, change2: Change) -> bool:
        if change1.type == 'removed':
            return self.can_merge_removals(change1, change2)
        elif change1.type == 'added':
            return self.can_merge_additions(change1, change2)
        return False
    
    def can_merge_removals(self, change1: Change, change2: Change) -> bool:
        if change1.old_index is not None and change2.old_index is not None:
            last_index = change1.old_index
            if hasattr(change1, 'content') and isinstance(change1.content, list):
                last_index = change1.old_index + len(change1.content) - 1
            return change2.old_index == last_index + 1
        return False
    
    def can_merge_additions(self, change1: Change, change2: Change) -> bool:
        if change1.new_index is not None and change2.new_index is not None:
            last_index = change1.new_index
            if hasattr(change1, 'content') and isinstance(change1.content, list):
                last_index = change1.new_index + len(change1.content) - 1
            return change2.new_index == last_index + 1
        return False
    
    def merge_changes(self, target: Change, source: Change):
        if hasattr(target, 'content') and hasattr(source, 'content'):
            if isinstance(target.content, list):
                if isinstance(source.content, list):
                    target.content.extend(source.content)
                else:
                    target.content.append(source.content)
            else:
                target.content = [target.content]
                if isinstance(source.content, list):
                    target.content.extend(source.content)
                else:
                    target.content.append(source.content)


class PDFDiffer:
    def __init__(self, similarity_threshold: float = 0.7, min_words_for_move: int = 8):
        self.similarity_threshold = similarity_threshold
        self.min_words_for_move = min_words_for_move
        self.extractor = BlockExtractor()
        self.organizer = ChangeOrganizer()
    
    def compare(self, pdf1_data: bytes, pdf2_data: bytes) -> DiffResult:
        blocks1 = self.extractor.extract_blocks_from_pdf(pdf1_data)
        blocks2 = self.extractor.extract_blocks_from_pdf(pdf2_data)
        
        detector = ChangeDetector(
            blocks1, blocks2, 
            self.similarity_threshold, 
            self.min_words_for_move
        )
        
        changes = detector.detect_changes()
        organized_changes = self.organizer.organize_changes(changes)
        
        statistics = self.calculate_statistics(organized_changes, len(blocks1), len(blocks2))
        metadata = self.create_metadata()
        
        return DiffResult(
            changes=organized_changes,
            statistics=statistics,
            metadata=metadata
        )
    
    def calculate_statistics(self, changes: List[Change], total_old: int, total_new: int) -> Statistics:
        stats = Statistics(
            total_blocks_old=total_old,
            total_blocks_new=total_new
        )
        
        for change in changes:
            if change.type == 'unchanged':
                stats.unchanged += 1
            elif change.type == 'modified':
                stats.modified += 1
            elif change.type == 'added':
                stats.added += 1
            elif change.type == 'removed':
                stats.removed += 1
            elif change.type == 'moved':
                stats.moved += 1
            elif change.type == 'moved_and_modified':
                stats.moved_and_modified += 1
        
        return stats
    
    def create_metadata(self) -> Metadata:
        return Metadata(
            generated_at=datetime.now().isoformat(),
            diff_version='1.0',
            similarity_threshold=self.similarity_threshold
        )
    
    def to_json(self, result: DiffResult) -> str:
        output = {
            'metadata': {
                'generated_at': result.metadata.generated_at,
                'diff_version': result.metadata.diff_version,
                'similarity_threshold': result.metadata.similarity_threshold
            },
            'statistics': {
                'total_blocks_old': result.statistics.total_blocks_old,
                'total_blocks_new': result.statistics.total_blocks_new,
                'unchanged': result.statistics.unchanged,
                'modified': result.statistics.modified,
                'added': result.statistics.added,
                'removed': result.statistics.removed,
                'moved': result.statistics.moved,
                'moved_and_modified': result.statistics.moved_and_modified
            },
            'changes': [change.to_dict() for change in result.changes]
        }
        
        return json.dumps(output, indent=2, default=str)