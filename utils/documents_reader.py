# Import basic libraries
from llama_index.readers.file import PandasExcelReader
from llama_index.core import Document
from docx import Document as DocxDocument
from paddleocr import PaddleOCR
import fitz
import os
import pandas as pd
import nest_asyncio
nest_asyncio.apply()


def read_documents(path: str) -> list:
    ocr = PaddleOCR(use_angle_cls=True, lang='id', show_log=False)
    reader = PandasExcelReader()

    docs_all = []

    document_path = path
    df_metadata = pd.read_csv('./files_metadata.csv')

    for file in os.listdir(document_path):
        file_path = os.path.join(document_path, file)
        file_metadata = df_metadata[df_metadata['new_filename'] == file]
        if file.endswith('.pdf'):
            doc = fitz.open(file_path)
            text = ''
            for page in doc:
                text += page.get_text()

            if not text.strip():
                ocr_result = ocr.ocr(file_path)
                for result in ocr_result:
                    for line in result:
                        text += str(line[1][0]) + '\n'
            text = text.replace('\n', ' ')

            document = Document(
                text=text,
                metadata={
                    "file_name": file_metadata['new_filename'].values[0],
                    "title": file_metadata['title'].values[0],
                    "sector": file_metadata['sektor'].values[0],
                    "subsector": file_metadata['subsektor'].values[0],
                    "regulation_type": file_metadata['jenis_regulasi'].values[0],
                    "regulation_number": file_metadata['nomor_regulasi'].values[0],
                    "effective_date": file_metadata['tanggal_berlaku'].values[0],
                }
            )

            docs_all.append(document)

        elif file.endswith('.xlsm') or file.endswith('.xlsx'):
            # Handle for Excel (xlsm, xlsx)
            doc = reader.load_data(file_path)
            text = ''
            for sheet in doc:
                for row in sheet:
                    for cell in row:
                        text += str(cell) + ' '

            document = Document(
                text=text,
                metadata={
                    "file_name": file_metadata['new_filename'].values[0],
                    "title": file_metadata['title'].values[0],
                    "sector": file_metadata['sektor'].values[0],
                    "subsector": file_metadata['subsektor'].values[0],
                    "regulation_type": file_metadata['jenis_regulasi'].values[0],
                    "regulation_number": file
                    ['nomor_regulasi'].values[0],
                    "effective_date": file_metadata['tanggal_berlaku'].values[0],
                }
            )

            text = text.replace('\n', ' ')

            docs_all.append(document)

        elif file.endswith('.docx'):
            # Handle for Word (docx)
            doc = DocxDocument(file_path)
            text = ''
            for para in doc.paragraphs:
                text += para.text + '\n'

            text = text.replace('\n', ' ')

            document = Document(
                text=text,
                metadata={
                    "file_name": file_metadata['new_filename'].values[0],
                    "title": file_metadata['title'].values[0],
                    "sector": file_metadata['sektor'].values[0],
                    "subsector": file_metadata['subsektor'].values[0],
                    "regulation_type": file_metadata['jenis_regulasi'].values[0],
                    "regulation_number": file_metadata['nomor_regulasi'].values[0],
                    "effective_date": file_metadata['tanggal_berlaku'].values[0],
                }
            )

            docs_all.append(document)
    print(f"Read {len(docs_all)} documents")

    return docs_all
