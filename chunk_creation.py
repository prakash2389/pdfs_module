import pdfplumber
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import os
from Extract_Header_Footer_v2 import Footer_Extraction
from extract_unstructuredtable import is_table_exist_in_page, extract_table_from_unstructured
from langchain.text_splitter import CharacterTextSplitter
# from logger import get_logger
# log = get_logger()

from PyPDF2 import PdfReader
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import (
    OnlinePDFLoader,
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    PDFPlumberLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    PDFMinerLoader,
    OnlinePDFLoader
)
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (UnstructuredPDFLoader, {"mode" :"elements"}),
    # ".pdf": (UnstructuredPDFLoader,{} ),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}
def replacetextwithtable(text_string, table_string, table_markdown_string):
    integer_vector = list(range(len(text_string)))
    text_string_vector = list(text_string)
    filtered_text_vector = [char for char in text_string_vector if char not in [' ', '\n']]
    filtered_integer_vector = [integer_vector[i] for i, char in enumerate(text_string_vector) if char not in [' ', '\n']]
    text_string_cleaned = ''.join(filtered_text_vector)
    table_string_cleaned = table_string.replace("\n","").replace(" ","")
    first = text_string_cleaned.find(table_string_cleaned)
    last = first + len(table_string_cleaned)
    if len(filtered_integer_vector)==last:
        next_string = ''
    else:
        next_string = text_string[filtered_integer_vector[last]:]
    final_string = text_string[:filtered_integer_vector[first]] + '\n' + table_markdown_string +  '\n' + next_string
    return final_string
def textonlypage(documents, pagenum):
    string_current_page = []
    coorindates_current_page = []
    for ele in documents:
        if (ele.metadata["page_number"]-1) == pagenum:
            string_current_page.append(ele.page_content)
            ele.metadata['coordinates']['page_number'] = ele.metadata['page_number']
            coorindates_current_page.append(ele.metadata['coordinates'])
    return string_current_page, coorindates_current_page
def table_text_page(documents, pagenum, current_page_tables, current_page_tables_cood):
    arr = []
    table_array = []
    string_current_page = []
    coorindates_current_page = []
    for ele in documents:
        if (ele.metadata["page_number"] - 1) == pagenum:
            j = 0
            for table_index in range(len(current_page_tables)):
                box = ele.metadata["coordinates"]["points"]
                table_box = current_page_tables_cood[table_index]
                is_inside = table_is_inside(table_box, box)
                if is_inside:
                    j = j + 1
                    if table_index not in arr:
                        markdown_table = markdowntable(current_page_tables[table_index])
                        # markdown_table = markdown_table_to_normal(markdown_table)
                        string_current_page.append(markdown_table)
                        table_array.append(is_inside)
                        ele.metadata['coordinates']['page_number'] = ele.metadata['page_number']
                        coorindates_current_page.append(ele.metadata['coordinates'])
                        arr.append(table_index)
            if j == 0:
                string_current_page.append(ele.page_content)
                ele.metadata['coordinates']['page_number'] = ele.metadata['page_number']
                coorindates_current_page.append(ele.metadata['coordinates'])
                table_array.append(is_inside)
    return string_current_page, table_array, coorindates_current_page
def tables_extract(pdffromplumber, pagenum):
    current_page_tables = []
    current_page_tables_cood = []
    page = pdffromplumber.pages[pagenum]
    tables = page.find_tables()
    table_bboxes = [i.bbox for i in tables]
    page_tables = page.extract_tables()
    current_page_tables.extend(page_tables)
    current_page_tables_cood.extend(table_bboxes)
    return current_page_tables, current_page_tables_cood
def table_is_inside( table_box, box):
    box_x_min = min(point[0] for point in box)
    box_x_max = max(point[0] for point in box)
    box_y_min = min(point[1] for point in box)
    box_y_max = max(point[1] for point in box)
    table_x_min, table_y_min, table_x_max, table_y_max = table_box
    # Check if the box is inside the table_box
    is_inside = (box_x_min >= table_x_min - 3) and (box_x_max <= table_x_max + 3) and (
            box_y_min >= table_y_min - 3) and (box_y_max <= table_y_max + 3)
    return is_inside
def markdowntable(table):
    text = ""
    for row in table:
        for cell in row:
            if cell is not None:
                text += cell
    df = pd.DataFrame(table)
    markdown_tab = df.to_markdown(index=False)
    first = markdown_tab.find("\n")
    markdown_tab = markdown_tab[first:]
    return markdown_tab
def footersheaders(documents, pdf):
    k=[]
    for lst in [0,1,2,3,-1,-2,-3,-4]:
        a,b = [], []
        for pagenum in range(len(pdf.pages)):
            string = []
            page = []
            for ele in documents:
                if (ele.metadata["page_number"] - 1) == pagenum:
                    string.append(ele.page_content)
                    page.append(ele.page_content.lower().replace(" ", "").startswith('page'))
            if len(string) >= 4:
                a.append(string[lst])
                b.append(page[lst])
        if len(list(set(a))) == 1:
            k.append(lst)
        if b.count(False) == 0:
            k.append(lst)
    return k
def remove_footers(documents, pdf):
    elements = footersheaders(documents, pdf)
    for pagenum in range(len(pdf.pages)):
        m = 0
        for ele in documents:
            if (ele.metadata["page_number"] - 1) == pagenum:
                m = m + 1
        n = 0
        for ele in documents:
            if (ele.metadata["page_number"] - 1) == pagenum:
                ele.metadata['footer_logic'] = False
                for x in elements:
                    if (x >= 0) and (n == x):
                        ele.metadata['footer_logic'] = True
                    if (x < 0) and (n == m+x):
                        ele.metadata['footer_logic'] = True
                n = n + 1
    return documents
def textonlypage_previous(documents, pagenum, overlap):
    previous_string = ""
    coorindates_previous_string = []
    string_previous_page, coorindates_current_page = textonlypage(documents, pagenum)
    for lst_ele in np.arange(-1, -len(string_previous_page)-1, -1):
        if (len("".join(string_previous_page[lst_ele:])) > overlap):
            previous_string = "\n\n".join(string_previous_page[lst_ele:])
            coorindates_previous_string.append(coorindates_current_page[lst_ele:])
            break
    return previous_string, coorindates_previous_string
def texttablepage_previous(string_previous_page, table_array, coorindates_current_page, overlap):
    previous_string = ""
    coorindates_previous_string = []
    for lst_ele in np.arange(-1, -len(string_previous_page)-1, -1):
        if (len("".join(string_previous_page[lst_ele:])) > overlap) or table_array[lst_ele]==True:
            if table_array[lst_ele]==False:
                previous_string = "\n\n".join(string_previous_page[lst_ele:])
                coorindates_previous_string.append(coorindates_current_page[lst_ele:])
                break
    if previous_string is None:
        previous_string = "\n\n".join(string_previous_page)
        coorindates_previous_string = coorindates_current_page
    return previous_string, coorindates_previous_string
def previous_page_string(documents, pdf, pagenum, unstructured_table_flag):
    print(f"Started extracting previous_page_string")
    previous_page_tables, previous_page_tables_cood = tables_extract(pdf, pagenum - 1)
    if previous_page_tables == [] and unstructured_table_flag:
        if is_table_exist_in_page(pdf, documents, pagenum - 1):
            previous_page_tables, previous_page_tables_cood = extract_table_from_unstructured(pdf, pagenum - 1)
    if previous_page_tables == []:
        previous_string, coorindates_previous_string = textonlypage_previous(documents, pagenum - 1, 400)
    else:
        string_previous_page, table_previous_array, coorindates_current_page = table_text_page(documents, pagenum - 1, previous_page_tables,
                                                            previous_page_tables_cood)
        previous_string, coorindates_previous_string = texttablepage_previous(string_previous_page, table_previous_array, coorindates_current_page, 400)
    print(f"Completed extracting previous_page_string")
    return previous_string, coorindates_previous_string
def markdown_table_to_normal(markdown_table):
    # Convert Markdown table to regular table
    table = tabulate([], tablefmt="pipe")  # Create an empty table with pipe format
    table_lines = table.split("\n")  # Split the empty table into lines
    # Parse and append the Markdown table rows
    for line in markdown_table.strip().split("\n"):
        row = line.strip("|").split("|")
        table_lines.append(row)
    # print(table_lines)
    # Print the regular table
    regular_table = "\n".join(table_lines)
    return regular_table
def footer_addition_documents(documents, final_footer_output):
    for ele in documents:
        box = ele.metadata['coordinates']['points']
        page_number = ele.metadata['page_number']
        box_x_min = min(point[0] for point in box)
        box_x_max = max(point[0] for point in box)
        box_y_min = min(point[1] for point in box)
        box_y_max = max(point[1] for point in box)
        is_inside = False
        try:
            f = final_footer_output[page_number]
            for j in f:
                table_x_min, table_y_min, table_x_max, table_y_max = (j['x0'], j['y0'], j['x1'], j['y1'])
                is_inside = (box_x_min >= table_x_min - 3) and (box_x_max <= table_x_max + 3) and \
                            (box_y_min >= table_y_max*(ele.metadata['coordinates']['layout_height']/j['page_height']) - 3) and \
                            (box_y_max <= table_y_min*(ele.metadata['coordinates']['layout_height']/j['page_height']) + 3)
                if is_inside==True:
                    break
        except:
            is_inside = False
        ele.metadata['footer_check'] = is_inside
    return documents
def delete_footer_documents(documents, footertype):
    m = 0
    k = []
    for ele in documents:
        if ele.metadata[footertype]==True:
            k.append(m)
        m = m + 1
    # Indices to delete
    indices_to_delete = k
    # Sort indices in descending order to avoid index shifting issues
    indices_to_delete.sort(reverse=True)
    # Delete elements at specified indices
    for index in indices_to_delete:
        if 0 <= index < len(documents):
            del documents[index]
    return documents
def standard_chunk_creation(path_or_url, project_name, mongo_id, source_app):
    print(f"Standard chunk creation started")
    loader_class, loader_args = (UnstructuredPDFLoader, {})
    loader = loader_class(path_or_url, **loader_args)
    documents = loader.load()
    for i in range(0, len(documents)):
        documents[i].metadata['project'] = project_name.lower()
        documents[i].metadata['mongoid'] = mongo_id
        documents[i].metadata['sourceapp'] = source_app
    chunk_size = 2000
    chunk_overlap = 200
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                          separator='\n')
    texts = text_splitter.split_documents(documents)
    try:
        texts1, documents = standard_chunk_coordinates(path_or_url, texts, mongo_id=mongo_id, source_app=source_app, project_name=project_name)
        texts1 = add_start_end_points_to_texts_v2(texts1, documents)
    except:
        for i in range(len(texts)):
            texts[i].metadata["project"] = project_name.lower()
            texts[i].metadata["mongoid"] = mongo_id
            texts[i].metadata["sourceapp"] = source_app
            texts[i].metadata["pageIndex"] = False
            texts[i].metadata["left"] = False
            texts[i].metadata["width"] = False
            texts[i].metadata["top"] = False
            texts[i].metadata["height"] = False
        texts1 = texts
    print(f"Standard chunk creation completed")
    return texts1
def add_start_end_points_to_texts_v2(texts, documents):
    m = {}
    for i in range(len(texts)):
        if isinstance(texts[i].metadata['coorindates_current_page'][0], dict):
            start_points = texts[i].metadata['coorindates_current_page'][0]['points']
            start_page = texts[i].metadata['coorindates_current_page'][0]['page_number']
            layout_width = texts[i].metadata['coorindates_current_page'][0]['layout_width']
            layout_height = texts[i].metadata['coorindates_current_page'][0]['layout_height']
        else:
            start_points = texts[i].metadata['coorindates_current_page'][0][0]['points']
            start_page = texts[i].metadata['coorindates_current_page'][0][0]['page_number']
            layout_width = texts[i].metadata['coorindates_current_page'][0][0]['layout_width']
            layout_height = texts[i].metadata['coorindates_current_page'][0][0]['layout_height']
        if isinstance(texts[i].metadata['coorindates_current_page'][-1], dict):
            end_points = texts[i].metadata['coorindates_current_page'][-1]['points']
            end_page = texts[i].metadata['coorindates_current_page'][-1]['page_number']
        else:
            end_points = texts[i].metadata['coorindates_current_page'][-1][-1]['points']
            end_page = texts[i].metadata['coorindates_current_page'][-1][-1]['page_number']
        texts[i].metadata['pages'] = {}
        pages_iter = []
        for j in range(len(texts[i].metadata['coorindates_current_page'])):
            if isinstance(texts[i].metadata['coorindates_current_page'][j], dict):
                pages_iter.append(texts[i].metadata['coorindates_current_page'][j]['page_number'])
            else:
                for k in range(len(texts[i].metadata['coorindates_current_page'][j])):
                    pages_iter.append(texts[i].metadata['coorindates_current_page'][j][k]['page_number'])
        pages_iter = list(set(pages_iter))
        for page in pages_iter:
            # print(page)
            texts[i].metadata['pages'][page] = {}
            if page == start_page:
                # texts[i].metadata['pages'][page]['start_points'] = start_points
                texts[i].metadata['pages'][page]['start_points'] = start_points
            else:
                texts[i].metadata['pages'][page]['start_points'] = ((0, 0), (0, 0), (0, 0), (0, 0))
            if page == end_page:
                texts[i].metadata['pages'][page]['end_points'] = end_points
            else:
                texts[i].metadata['pages'][page]['end_points'] = ((layout_width, layout_height), (layout_width, layout_height), (layout_width, layout_height), (layout_width, layout_height))
            texts[i].metadata['pages'][page]['points'] = ((0, texts[i].metadata['pages'][page]['start_points'][0][1]),
                                                          (0, texts[i].metadata['pages'][page]['start_points'] [1][1]),
                                                          (layout_width, texts[i].metadata['pages'][page]['end_points'][2][1]),
                                                          (layout_width, texts[i].metadata['pages'][page]['end_points'][3][1]))
            texts[i].metadata['pages'][page].pop("start_points")
            texts[i].metadata['pages'][page].pop("end_points")
            texts[i].metadata['pages'][page]['layout_width'] = layout_width
            texts[i].metadata['pages'][page]['layout_height'] = layout_height
        texts[i].metadata.pop("coorindates_current_page")
        # texts[i].metadata.pop("pages")
        texts[i].metadata.pop("page")
        # texts[i].metadata['page'] = list(texts[i].metadata['pages'].keys())
        m[i] = texts[i].metadata['pages']
    texts = normalize_points(texts)
    texts = normalized_points_to_texts(texts, documents)
    return texts
def normalize_points(texts):
    for i in range(len(texts)):
        for j in texts[i].metadata['pages'].keys():
            box = texts[i].metadata['pages'][j]['points']
            left = min(point[0] for point in box)/texts[i].metadata['pages'][j]['layout_width'] * 100
            right = max(point[0] for point in box)/texts[i].metadata['pages'][j]['layout_width'] * 100
            top = min(point[1] for point in box)/texts[i].metadata['pages'][j]['layout_height'] * 100
            bottom = max(point[1] for point in box)/texts[i].metadata['pages'][j]['layout_height'] * 100
            texts[i].metadata['pages'][j]['normalized_points'] = {"pageIndex": j, "left": left, "width": right-left, "top": top, "height": bottom-top}
            texts[i].metadata['pages'][j].pop("points")
            texts[i].metadata['pages'][j].pop("layout_width")
            texts[i].metadata['pages'][j].pop("layout_height")
    return texts
def normalized_points_to_texts(texts, documents):
    for i in range(len(texts)):
        x = ['','','','','']
        for j in texts[i].metadata['pages'].keys():
            left=[]
            right=[]
            top=[]
            bottom=[]
            for ele in documents:
                if (ele.metadata["page_number"]) == j:
                    box = ele.metadata["coordinates"]["points"]
                    # print(box)
                    left.append(min(point[0] for point in box)/ele.metadata["coordinates"]['layout_width']*100)
                    right.append(max(point[0] for point in box)/ele.metadata["coordinates"]['layout_width']*100)
                    top.append(min(point[1] for point in box)/ele.metadata["coordinates"]['layout_height']*100)
                    bottom.append(max(point[1] for point in box)/ele.metadata["coordinates"]['layout_height']*100)
            if texts[i].metadata['pages'][j]['normalized_points']["left"] < min(left):
                l = min(left)
            else:
                l = texts[i].metadata['pages'][j]['normalized_points']["left"]
            if texts[i].metadata['pages'][j]['normalized_points']["width"] > max(right)-l:
                w = max(right)-l
            else:
                w = texts[i].metadata['pages'][j]['normalized_points']["width"]
            if texts[i].metadata['pages'][j]['normalized_points']["top"] < min(top):
                t = min(top)
            else:
                t = texts[i].metadata['pages'][j]['normalized_points']["top"]
            if texts[i].metadata['pages'][j]['normalized_points']["height"] > max(bottom)-t:
                h = max(bottom)-t
            else:
                h = texts[i].metadata['pages'][j]['normalized_points']["height"]
            # texts[i].metadata['pages'][j]['normalized_points'] = {'pageIndex': j-1, 'left': int(max(0,l-1)), 'width': int(min(100,w+4)), 'top': int(max(0,t-1)), 'height': int(min(100,h+4))}
            x[0] += "," + str(j-1)
            x[1] += "," + str(int(max(0, l-1)))
            x[2] += "," + str(int(min(100, w+4)))
            x[3] += "," + str(int(max(0, t-1)))
            x[4] += "," + str(int(min(100, h+4)))
        texts[i].metadata["pageIndex"] = x[0][1:]
        texts[i].metadata["left"] = x[1][1:]
        texts[i].metadata["width"] = x[2][1:]
        texts[i].metadata["top"] = x[3][1:]
        texts[i].metadata["height"] = x[4][1:]
        texts[i].metadata.pop("pages")
    return texts
def plot_point_chunk(texts, pdf):
    for i in range(len(texts)):
        for j in texts[i].metadata['pages'].keys():
            # Load the input image
            image = pdf.pages[j-1].to_image(resolution=100)
            file_path = "output_image.png"
            image.save(file_path)
            input_image = cv2.imread('output_image.png')  # Replace 'input_image.jpg' with the path to your input image
            # start_points = list(texts[i].metadata['pages'][j]['start_points'])
            # end_points = list(texts[i].metadata['pages'][j]['end_points'])
            points = list(texts[i].metadata['pages'][j]['points'])
            layout_width = texts[i].metadata['pages'][j]['layout_width']
            layout_height = texts[i].metadata['pages'][j]['layout_height']
            # a = 0*(input_image.shape[1])/layout_width
            # b = start_points[0][1]*(input_image.shape[0])/layout_height
            # c = 0*(input_image.shape[1])/layout_width
            # d = start_points[1][1]*(input_image.shape[0])/layout_height
            # e = layout_width*(input_image.shape[1])/layout_width
            # f = end_points[2][1]*(input_image.shape[0])/layout_height
            # g = layout_width*(input_image.shape[1])/layout_width
            # h = end_points[3][1]*(input_image.shape[0])/layout_height

            a = points[0][0]*(input_image.shape[1])/layout_width
            b = points[0][1]*(input_image.shape[0])/layout_height
            c = points[1][0]*(input_image.shape[1])/layout_width
            d = points[1][1]*(input_image.shape[0])/layout_height
            e = points[2][0]*(input_image.shape[1])/layout_width
            f = points[2][1]*(input_image.shape[0])/layout_height
            g = points[3][0]*(input_image.shape[1])/layout_width
            h = points[3][1]*(input_image.shape[0])/layout_height

            points = [(a, b), (c, d), (e, f), (g, h)]
            # Extract x and y coordinates from the points
            x_values, y_values = zip(*points)

            # Find minimum and maximum values for x and y coordinates
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)

            # Draw the rectangular box on the input image
            cv2.rectangle(input_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 5)

            # Annotate corners with labels
            corner_labels = {'A': (x_min, y_min), 'B': (x_min, y_max), 'C': (x_max, y_max), 'D': (x_max, y_min)}

            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (0, 0, 0)
            cv2.imwrite('plots\output_image' + str(i) + 'page' + str(j) + '.png', input_image)  # Save the image as 'output_image.jpg'
def standard_chunk_coordinates(path_or_url, texts, mongo_id, source_app, project_name):
    loader_class, loader_args = (UnstructuredPDFLoader, {"mode" : "elements"})
    loader = loader_class(path_or_url, **loader_args)
    documents1 = loader.load()
    m = '\n\n'.join(doc.page_content for doc in documents1)
    position=[]
    for i in range(len(texts)):
        start_position = m.replace('\n', '').find(texts[i].page_content.replace('\n', ''))
        end_position = start_position + len(texts[i].page_content.replace('\n', ''))
        position.append((start_position, end_position))

    n = ''
    index=[]
    for i in range(len(documents1)):
        index.append(0 if i == 0 else len(n) - 1)
        n = n + documents1[i].page_content.replace('\n', '')

    for i in range(len(texts)):
        if i==0:
            start= 0
            end=index.index(position[i][1] - 1)
        elif i!=(len(position)-1):
            start= index.index(position[i][0]-1)
            end= index.index(position[i][1]-1)
        elif i==(len(position)-1):
            start= index.index(position[i][0] - 1)
            end= len(index)-1
        doc_temp = documents1[start:end]
        coorindates_current_page = []
        for j in range(len(doc_temp)):
            doc_temp[j].metadata['coordinates']['page_number'] = doc_temp[j].metadata['page_number']
            coorindates_current_page.append(doc_temp[j].metadata['coordinates'])
        texts[i].metadata = {'project': project_name.lower(), 'page': ",".join(list(set([str(doc_temp[i].metadata['page_number']) for i in range(len(doc_temp))]))), 'mongoid': mongo_id, 'sourceapp': source_app, 'coorindates_current_page': coorindates_current_page}
    return texts, documents1
def chunk_pdf(pdf_path, documents, project_name, mongo_id, source_app, standard_chunk = False, unstructured_table_flag=False, footercheck = True, footerlogic = True):
    if standard_chunk == False:
        try:
            print(f"pdf plumber reading.. ")
            pdf = pdfplumber.open(pdf_path)
            print(f"Footer_Extraction started.. ")
            final_footer_output = Footer_Extraction(pdf_path)
            print(f"Footer added to documents.. ")
            documents = remove_footers(documents, pdf)
            documents = footer_addition_documents(documents, final_footer_output)
            if footercheck ==True:
                documents = delete_footer_documents(documents, "footer_check")
            if footerlogic ==True:
                documents = delete_footer_documents(documents, "footer_logic")
            text = ""
            m = 0
            texts = []
            table_info = []
            table_header = ''
            for i in range(len(pdf.pages)):
                print(i)
                print(f"########### Page {i + 1} ############")
                string_current_page = []
                previous_string = ""
                coorindates_previous_string = []
                print(f"Started table extraction from pdf plumber")
                current_page_tables, current_page_tables_cood = tables_extract(pdf, i)
                if current_page_tables == [] and unstructured_table_flag:
                    if is_table_exist_in_page(pdf, documents, i):
                        print(f"Started unstructured table extraction")
                        current_page_tables, current_page_tables_cood = extract_table_from_unstructured(pdf, i)
                        print(f"Completed unstructured table extraction")
                if current_page_tables == []:
                    string_current_page, coorindates_current_page = textonlypage(documents, i)
                    # if (i > 0): previous_string, coorindates_previous_string = previous_page_string(documents, pdf, i, unstructured_table_flag)
                    text_current_page = "\n" + previous_string + "\n\n" + "\n\n".join(string_current_page)
                    texts.append(Document(page_content=text_current_page, metadata={'project': project_name.lower(),'page': str(i), 'mongoid': mongo_id, 'sourceapp': source_app, 'coorindates_current_page': coorindates_previous_string + coorindates_current_page}))
                else:
                    string_current_page, table_current_array, coorindates_current_page = table_text_page(documents, i, current_page_tables, current_page_tables_cood)
                    # if i>0: previous_string, coorindates_previous_string = previous_page_string(documents, pdf, i, unstructured_table_flag)
                    # if table_current_array!=[]:
                    alltables = table_current_array.count(False) == 0
                    # if (table_current_array[-1] == True) and (table_current_array.count(False)>0):
                    #     revers = table_current_array[::-1]
                    #     temp = (len(string_current_page)-revers.index(False))
                    #     table_info = string_current_page[(temp-2):(temp)]
                    #     table_info_coords = coorindates_current_page[(temp-2):(temp)]
                    #     table_header = markdowntable(current_page_tables[table_current_array[0:temp+1].count(True)-1][0:3])
                    #     table_header_coords = coorindates_current_page[table_current_array[0:temp+1].count(True)-1:]
                    #     string_current_page = string_current_page[:(temp - 1)]
                    #     table_current_array = table_current_array[:(temp - 1)]
                    #     coorindates_current_page = coorindates_current_page[:(temp - 1)]
                    text = text + "\n\n".join(string_current_page)
                    texts.append(Document(page_content=text, metadata={'project': project_name.lower(), 'page': str(i), 'mongoid': mongo_id, 'sourceapp': source_app, 'coorindates_current_page': ""}))
                    text = ""
                        # alltables = table_current_array.count(False) == 0
                        # if alltables:
                        #     m += 1
                        #     if (m > 1):
                        #         text = text + "\n" + "\n\n".join(table_info) + "\n\n" + table_header + "\n" + "\n\n".join(string_current_page)
                        #         texts.append(Document(page_content=text, metadata={'project': project_name.lower(), 'page': str(i), 'mongoid': mongo_id, 'sourceapp': source_app, 'coorindates_current_page': table_info_coords + table_header_coords + coorindates_current_page}))
                        #         text = ""
                        #     else:
                        #         text = text + "\n" + previous_string + "\n\n" + "\n\n".join(string_current_page)
                        #         texts.append(Document(page_content=text, metadata={'project': project_name.lower(), 'page': str(i), 'mongoid': mongo_id, 'sourceapp': source_app, 'coorindates_current_page': coorindates_previous_string + coorindates_current_page}))
                        #         text = ""
                        # else:
                        #     if m > 0:
                        #         text = text + "\n" + "\n\n".join(table_info) + "\n\n" + table_header + "\n" + "\n\n".join(string_current_page)
                        #         texts.append(Document(page_content=text, metadata={'project': project_name.lower(), 'page': str(i), 'mongoid': mongo_id, 'sourceapp': source_app, 'coorindates_current_page': table_info_coords + table_header_coords + coorindates_current_page}))
                        #     else:
                        #         text = text + "\n\n" + previous_string + "\n\n" + "\n\n".join(string_current_page)
                        #         texts.append(Document(page_content=text, metadata={'project': project_name.lower(), 'page': str(i), 'mongoid': mongo_id, 'sourceapp': source_app, 'coorindates_current_page': coorindates_previous_string + coorindates_current_page}))
                        #     text = ""
                        #     m = 0
            pdf.close()
            try:
                texts = add_start_end_points_to_texts_v2(texts, documents)
            except:
                for i in range(len(texts)):
                    texts[i].metadata["project"] = project_name.lower()
                    texts[i].metadata["mongoid"] = mongo_id
                    texts[i].metadata["sourceapp"] = source_app
                    texts[i].metadata["pageIndex"] = False
                    texts[i].metadata["left"] = False
                    texts[i].metadata["width"] = False
                    texts[i].metadata["top"] = False
                    texts[i].metadata["height"] = False
            # plot_point_chunk(texts, pdf)
        except:
            print(f"Error occured in custom chunk")
            texts = standard_chunk_creation(pdf_path, project_name, mongo_id, source_app)
    if standard_chunk == True:
        print(f"Standard chunk creation started")
        texts = standard_chunk_creation(pdf_path, project_name, mongo_id, source_app)
    return texts
