import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from docx import Document

from hybridrag.prepare import clean_text, extract_text


class CleanTextTest(unittest.TestCase):
    def test_drops_page_numbers_and_control_chars(self):
        raw = "第一条 借阅上限 10 册。\x00\n12\n第 2 页\n\n\n逾期按日计算。"
        self.assertEqual(clean_text(raw), "第一条 借阅上限 10 册。\n逾期按日计算。")


class DocxExtractTest(unittest.TestCase):
    def test_reads_paragraph_text(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rule.docx"
            document = Document()
            document.add_paragraph("本科生一次最多可以借 10 册。")
            document.save(path)
            self.assertIn("10 册", extract_text(path))


if __name__ == "__main__":
    unittest.main()
