from typing import List, Dict, Any
from pathlib import Path
import pdfplumber
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader

class FileParser:
    """ 文件解析： pdf、docx、txt、xlsx、pptx """

    @staticmethod
    def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
        result = []
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                result.append({"location": f"第{page_idx}页", "raw_text": text})
        return result

    @staticmethod
    def parse_docx(file_path: str) -> List[Dict[str, Any]]:
        doc = Document(file_path)
        result = []
        for para_idx, para in enumerate(doc.paragraphs, start=1):
            result.append({"location": f"Word段落：{para_idx}", "raw_text": para.text})
        return result

    @staticmethod
    def parse_txt(file_path: str) -> List[Dict[str, Any]]:
        result = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line_idx, line in enumerate(lines, start=1):
            result.append({"location": f"文本行：{line_idx}", "raw_text":line})
        return result

    @staticmethod
    def parse_xlsx(file_path: str) -> List[Dict[str, Any]]:
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        docs = loader.load()
        res = []
        for doc in docs:
            sheet_name = doc.metadata.get("sheet_name", "未知工作表")
            row_num = doc.metadata.get("row_number", -1)
            res.append({"location": f"工作表：{sheet_name}，表格行：{row_num}", "raw_text": doc.page_content})
        return res

    @staticmethod
    def parse_pptx(file_path: str) -> List[Dict[str, Any]]:
        loader = UnstructuredPowerPointLoader(file_path, mode="elements")
        docs = loader.load()
        res = []
        for d in docs:
            slide_num = d.metadata.get("page_number", -1)
            res.append({"location": f"第{slide_num}张幻灯片", "raw_text": d.page_content})
        return res

    @classmethod
    def parse(cls, file_path: str) -> List[Dict[str, Any]]:
        suffix = Path(file_path).suffix.lower()
        mapping = {
            ".pdf": cls.parse_pdf,
            ".docx": cls.parse_docx,
            ".pptx": cls.parse_pptx,
            ".txt": cls.parse_txt,
            ".xlsx": cls.parse_xlsx
        }
        if suffix not in mapping:
            raise Exception(f"暂不支持文件后缀{suffix}")
        return mapping[suffix](file_path)

class TextCleaner:
    """文本清洗：保留段落换行，清除多余空白、孤立页码行，修复pdf断行"""
    @staticmethod
    def clean_single_text(text: str) -> str:
        import re
        text = text.strip()
        # 连续多个换行替换成双换行，保留段落
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        lines = [line.rstrip() for line in text.splitlines()]
        content = "\n".join(lines)
        out_lines = []
        for ln in content.split("\n"):
            if ln.strip().isdigit() and len(ln) <= 4:
                continue
            out_lines.append(ln)
        content = "\n".join(out_lines)
        content = content.replace("-\n", "")
        return content

import re


class DocumentChunker:
    """
    文档分块模块：
    1. 优先按一级大标题（一、二、…；【xxx】）切分，尽量让同一个大标题下的内容归入同一块；
    2. 若大标题章节整体不超过 chunk_max_size，则整块输出；
    3. 若超过 chunk_max_size，先按二级标题（1.、2.1、2.1.1 等）细分，仍超长则用递归字符切分兜底。
    输入：FileParser输出的片段列表；doc_id、doc_name来自业务层
    输出：chunk列表，每一块附带meta溯源信息
    注意：因为全文合并再切分，如果本块来自多个原始片段，location标记为【跨多源片段】
    """
    # 一级标题：中文数字顿号（一、二、…、十一、十二、…）或中括号标题【xxx】
    _H1_RE = re.compile(r"^[一二三四五六七八九十百]{1,3}、|^【[^】]+】", re.MULTILINE)
    # 二级标题：多级数字编号（2.1、2.1.1）或单级数字编号（1.，且点后不是数字，避免把“2.1”截成“2.”）
    _H2_RE = re.compile(r"^\d+(?:\.\d+)+|^\d+\.(?!\d)", re.MULTILINE)

    def __init__(self, chunk_max_size: int = 500, chunk_overlap: int = 100):
        self.chunk_max_size = chunk_max_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_max_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "，", "、"]
        )

    @staticmethod
    def _line_end(text: str, pos: int) -> int:
        """返回 pos 所在行的行尾（不含换行符）的字符索引"""
        nl = text.find("\n", pos)
        return nl if nl != -1 else len(text)

    def split_by_title(self, full_text: str) -> List[Dict[str, Any]]:
        """
        按一级标题切分全文，返回大章节列表。
        每个章节包含：
        - title: 一级标题整行文本（如“一、系统总体介绍”），无标题时为 ""
        - content: 标题行之后的正文（不含一级标题行本身）
        - start/end: 整个章节（含标题行）在 full_text 中的字符区间 [start, end)
        - content_start: 正文在 full_text 中的起始字符偏移
        """
        matches = list(self._H1_RE.finditer(full_text))
        chapters: List[Dict[str, Any]] = []
        # 第一个一级标题之前的序言也保留为一个“无标题”章节
        if matches and matches[0].start() > 0:
            chapters.append({
                "title": "",
                "content": full_text[0:matches[0].start()],
                "start": 0,
                "end": matches[0].start(),
                "content_start": 0,
            })
        if not matches:
            chapters.append({
                "title": "",
                "content": full_text,
                "start": 0,
                "end": len(full_text),
                "content_start": 0,
            })
            return chapters
        for idx, m in enumerate(matches):
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
            title_end = self._line_end(full_text, start)
            title = full_text[start:title_end].strip()
            content_start = title_end + 1 if title_end < end else end
            # 跳过正文开头的换行，避免块文本出现多余空行
            while content_start < end and full_text[content_start] == "\n":
                content_start += 1
            chapters.append({
                "title": title,
                "content": full_text[content_start:end],
                "start": start,
                "end": end,
                "content_start": content_start,
            })
        return chapters

    def _split_by_h2(self, content: str) -> List[Dict[str, Any]]:
        """把大标题正文按二级标题细分，返回子节列表；子节的 start/end/content_start 相对 content 本身"""
        matches = list(self._H2_RE.finditer(content))
        sections: List[Dict[str, Any]] = []
        if not matches:
            return [{"title": "", "content": content, "start": 0, "end": len(content), "content_start": 0}]
        if matches[0].start() > 0:
            sections.append({
                "title": "",
                "content": content[0:matches[0].start()],
                "start": 0,
                "end": matches[0].start(),
                "content_start": 0,
            })
        for idx, m in enumerate(matches):
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            title_end = self._line_end(content, start)
            title = content[start:title_end].strip()
            content_start = title_end + 1 if title_end < end else end
            # 跳过正文开头的换行，避免块文本出现多余空行
            while content_start < end and content[content_start] == "\n":
                content_start += 1
            sections.append({
                "title": title,
                "content": content[content_start:end],
                "start": start,
                "end": end,
                "content_start": content_start,
            })
        return sections

    @staticmethod
    def _resolve_location(spans: List, start: int, end: int) -> str:
        """根据字符区间 [start, end)，反查它覆盖了哪些原始 fragment，输出精准位置标记"""
        hit = [loc for loc, s, e in spans if s < end and e > start]
        if not hit:
            return "未知位置"
        if len(hit) == 1:
            return hit[0]
        return f"跨多源片段：（{hit[0]}）至（{hit[-1]}）"

    def _build_meta(self, doc_id: str, doc_name: str, chapter_title: str,
                    sub_title: str, spans: List, start: int, end: int) -> Dict[str, Any]:
        meta = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chapter_title": chapter_title,
            "location": self._resolve_location(spans, start, end),
        }
        if sub_title:
            meta["sub_title"] = sub_title
        return meta

    def _make_block(self, chapter_title: str, sub_title: str, body: str) -> str:
        clean_title = (chapter_title or "").strip().strip("【】")
        prefix = f"【章节：{clean_title}】" if clean_title else "【章节：未命名】"
        if sub_title:
            prefix += f"\n【小节：{sub_title}】"
        if body:
            return f"{prefix}\n{body}"
        return prefix

    def _recursive_chunks(self, doc_id: str, doc_name: str, chapter_title: str,
                          sub_title: str, content: str, spans: List, base_offset: int) -> List[Dict]:
        """对超长正文做递归字符切分，返回 chunk 列表；base_offset 是 content 相对 full_text 的偏移"""
        chunks = []
        sub_list = self.splitter.split_text(content)
        cursor2 = 0
        for sub_txt in sub_list:
            pos = content.find(sub_txt, cursor2)
            if pos < 0:
                pos = cursor2
            cursor2 = pos + 1
            block_text = self._make_block(chapter_title, sub_title, sub_txt)
            meta = self._build_meta(
                doc_id, doc_name, chapter_title, sub_title,
                spans, base_offset + pos, base_offset + pos + len(sub_txt),
            )
            chunks.append({"text": block_text, "meta": meta})
        return chunks

    def chunk_document(self, parsed_fragments: List[Dict], doc_id: str, doc_name: str) -> List[Dict]:
        # 清洗并拼接全文，同时记录每个 fragment 在 full_text 中的字符区间
        cleaned_list = []
        spans = []  # [(location, start_char, end_char), ...]
        cursor = 0
        for frag in parsed_fragments:
            cleaned = TextCleaner.clean_single_text(frag["raw_text"])
            cleaned_list.append(cleaned)
            spans.append((frag["location"], cursor, cursor + len(cleaned)))
            cursor += len(cleaned) + 2  # +2 对应 join 用的 "\n\n"
        full_text = "\n\n".join(cleaned_list)

        final_chunks: List[Dict] = []

        for chap in self.split_by_title(full_text):
            # 跳过既无标题也无正文的空章节
            if not chap["title"] and not chap["content"].strip():
                continue
            chap_len = chap["end"] - chap["start"]
            if chap_len <= self.chunk_max_size:
                # 大标题章节整体不超长：整块输出
                block_text = self._make_block(chap["title"], "", chap["content"])
                meta = self._build_meta(
                    doc_id, doc_name, chap["title"], "",
                    spans, chap["start"], chap["end"],
                )
                final_chunks.append({"text": block_text, "meta": meta})
            else:
                # 超长：先按二级标题细分，尽量保持子标题语义完整
                children = self._split_by_h2(chap["content"])
                has_children = len(children) > 1 or bool(children[0]["title"])
                if not has_children:
                    # 没有二级标题，直接递归字符切分兜底
                    final_chunks.extend(self._recursive_chunks(
                        doc_id, doc_name, chap["title"], "", chap["content"],
                        spans, chap["content_start"],
                    ))
                else:
                    for child in children:
                        child_len = child["end"] - child["start"]
                        abs_start = chap["content_start"] + child["start"]
                        abs_end = chap["content_start"] + child["end"]
                        if child_len <= self.chunk_max_size:
                            block_text = self._make_block(chap["title"], child["title"], child["content"])
                            meta = self._build_meta(
                                doc_id, doc_name, chap["title"], child["title"],
                                spans, abs_start, abs_end,
                            )
                            final_chunks.append({"text": block_text, "meta": meta})
                        else:
                            # 子标题章节仍超长：递归字符切分
                            child_base = chap["content_start"] + child["content_start"]
                            final_chunks.extend(self._recursive_chunks(
                                doc_id, doc_name, chap["title"], child["title"], child["content"],
                                spans, child_base,
                            ))
        return final_chunks