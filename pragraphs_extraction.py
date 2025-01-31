import os
import statistics
import pdfplumber as pdfplumber

pdf_path = r"D:\Downloads\CaseLaw_XML-20241206T060931Z-001\CaseLaw_XML\Caselaw-Sample-Text-pdf-files"
pdf_file = "SDWVA_5163482_XM.3.24-cr-00100.40.request.pdf"
file = os.path.join(pdf_path, pdf_file)
pdf = pdfplumber.open(file)


def extract_footer_paragraphs_from_pdfplumber(pdf):
    footerlist = []
    paragraph_list = []
    for page in pdf.pages:
        m = False
        footers=[]
        footernote = ""
        min_left = min([x["x0"] for x in page.extract_text_lines()])
        max_left = min([x["x1"] for x in page.extract_text_lines()])
        # print(min_left, max_left)
        gap = []
        for i in range(len(page.extract_text_lines())-1):
            gap.append(int(page.extract_text_lines()[i+1]["bottom"] - page.extract_text_lines()[i]["top"]))
        mode_gap = statistics.mode(gap)
        gap = [100] + gap
        para_text = ""
        line_id = 0
        for line_no in range(len(page.extract_text_lines())):
            next_id = True
        # for lines in page.extract_text_lines():
            lines = page.extract_text_lines()[line_no]
            try:
                next_lines = page.extract_text_lines()[line_no + 1]
            except:
                next_id = False
            if (lines['chars'][0]['y0'] < next_lines['chars'][0]['y1']) and next_id and (next_lines['chars'][0]['size'] > (lines['chars'][0]['size'] * 1.05)) and (lines['chars'][0]['text'].isdigit()):
                footernote = footernote + "\n\n" + lines['text']
                footernote = footernote + "\n\n" + next_lines['text']
                m = True
            else:
                if len(lines['chars'])>1:
                    if len(lines['chars'])>2:
                        k = 2
                    else:
                        k = 1
                    if m == True and (lines['chars'][k]['matrix'][5] == lines['chars'][0]['matrix'][5]) and (
                            lines['chars'][k]['size'] == lines['chars'][0]['size']):
                        if ("page" in lines['text'].lower()) or lines['text'].isdigit() or lines['text'].replace(" ","").replace("/","").replace("-","").isdigit():
                            pass
                        else:
                            footernote = footernote + " " + lines['text']
                    elif (lines['chars'][k]['matrix'][5] < lines['chars'][0]['matrix'][5]) and (
                            lines['chars'][k]['size'] > lines['chars'][0]['size']) and (lines['chars'][0]['text'].isdigit()):
                        footernote = footernote + "\n\n" + lines['text']
                        m = True
                    else:
                        line = page.extract_text_lines()[line_no]
                        indices = find_super_script(line)
                        text = replace_super_script(line, indices)
                        if (line["x0"] < (min_left + 2)) and (gap[line_id] <= (mode_gap * 1.1)):
                            para_text = para_text + text
                        else:
                            if para_text != "":
                                paragraph_list.append(para_text)
                                footerlist.append(0)
                            para_text = text
                        line_id = line_id + 1
                        m = False
        paragraph_list.append(para_text)
        footerlist.append(0)
        if footernote != "":
            paragraph_list.append(footernote.strip())
            footerlist.append(1)
    return paragraph_list, footerlist
def find_super_script(line):
    a = [x["size"] for x in line["chars"]]
    b = [x['matrix'][5] for x in line["chars"]]
    # Find indices where both conditions are met
    indices = []
    for i in range(1, len(a)):
        if a[i] < a[i - 1] - 1 and b[i] > b[i - 1] - 5:
            indices.append(i)
    return indices
def replace_super_script(line, indices):
    text = ""
    k=0
    for m in range(len(line["chars"])):
        while line["text"][k] != line["chars"][m]["text"]:
            text = text + line["text"][k]
            k = k + 1
        if m in indices:
            text = text + "//super//" +line["chars"][m]["text"] + "//super//"
        else:
            text = text + line["chars"][m]["text"]
        k = k + 1
    return text

paragraph_list, footerlist = extract_footer_paragraphs_from_pdfplumber(pdf)


for i in range(50):
    print(i)
    print(paragraph_list[i])
    print(footerlist[i])
    print("\n")




