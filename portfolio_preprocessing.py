import fitz
import re
import os
import cv2
import dlib
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from collections import defaultdict

# 이메일 주소, 전화번호, 교육기관명 패턴, 웹사이트 링크 패턴
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
phone_pattern = r"\b(?:010|011|016|017|018|019|10)(?:\s*\-\s*|\s+)\d{3,4}(?:\s*\-\s*|\s+)\d{4}\b"
edu_pattern = r"\b\S+대학교\b|\b\S+학교\b"
#url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
path_file = "/Users/jiho/desktop/user_list/"


# 전체 TLD 목록 (예시로 일부만 포함, 실제 사용 시에는 최신 목록을 사용)
tlds = (
    r"com|net|org|edu|gov|mil|int|co|us|uk|de|jp|fr|au|ca|cn|br|in|ru|pl|it|nl|se|no|fi|dk|ch|at|be|es|pt|gr|hk|kr|sg|tw|my|ph|za|mx|ar|cl|pe|uy|do|pa|cr|gt|hn|sv|jm|tt|ky|ai|lc|vc|ms|ws|io"
)

url_pattern = re.compile(r"""
    (?:http[s]?://)?                   # http:// 또는 https:// (선택 사항)
    (?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+  # 도메인 이름
    (?:""" + tlds + r""")            # TLD
    (?:\:[0-9]{1,5})?                  # 포트 번호 (선택 사항)
    (?:/[a-zA-Z0-9-._~:/?#@!$&'()*+,;=%]*)?  # URL 경로, 쿼리 문자열 및 프래그먼트 (선택 사항)
    """, re.VERBOSE)


print(f"""
| 포트폴리오 / pdf 마스킹 사용 설명서
|
| 1. 사용자 명을 입력하세요.
| ('사용자'는 user_list/'사용자') 입니다.
|
| 2. 추가로 마스킹 할 내용을 입력하세요
| (마스킹할 내용:개체 정보)
| ex1) 최지태 : name
| ex2) 가천 : univ
| ...
""")

try:
    # 폰트 경로를 정확히 지정하세요
    font = ImageFont.truetype("/Users/kieh/Library/Fonts/NanumBarunGothic.ttf", 30)
    print("* 나눔바른고딕 폰트를 사용합니다.\n")
except IOError:
    # 폰트가 없을 경우 기본 폰트 사용
    font = ImageFont.load_default()
    print("* 기본 폰트를 사용합니다.\n")

while True :
    user = input(f"| 사용자 명 입력 : \n>>> ")

    # PDF 파일 경로
    paths = []
    # 추가할 파일 경로
    user_path = user + '/' + user
    resume_path = path_file+user_path+"_이력서.pdf"
    if os.path.exists(resume_path):
        paths.append(resume_path)
    portfolio_path = path_file+user_path+"_포트폴리오.pdf"
    if os.path.exists(portfolio_path):
        paths.append(portfolio_path)
    if paths:
        break
    else:
        print(f"{user} 사용자의 파일이 존재하지 않습니다. 다시 입력해주세요.")

url_log_path = path_file+user_path+"_urls.txt"

# 검색할 텍스트 리스트 및 대체 텍스트 리스트 초기화
search_texts = []  # 추가하고 싶은 검색어를 여기에 추가하세요
replace_texts = []  # 검색어와 동일한 순서로 대체할 텍스트를 여기에 추가하세요
url_texts = []
masked_texts = [] # 마스킹 된 파일 저장
substitution_counts = defaultdict(int) # 마스킹 된 단어 카운팅

print("| (마스킹할 내용:개체 정보)를 입력하세요\n| * 형식에 맞지 않으면 종료됩니다 *\n ")

while True:
    input_text = input()
    
    if input_text == "q":
        break
    
    # 입력된 문자열이 '~~~:~~~' 형식인지 확인
    if ":" not in input_text:
        print("| 입력이 종료됩니다.\n")
        break
    
    # 올바른 형식이면 분리하여 리스트에 추가
    st, rt = input_text.split(":")
    search_texts.append(st)
    replace_texts.append('[' + rt + ']')

print(f"| ---------- 다음과 같은 정보들이 마스킹 됩니다. ---------- |\n| 기본값 :\nemail, phone, univ\n")


if search_texts and replace_texts:
    print("| 사용자 입력값 :\n")
    for st, rt in zip(search_texts, replace_texts):
        print(f"{st}:{rt[1:-1]}")

print("\n| ----------         Loading ...         ---------- |")

# 얼굴 감지 모델 로드 (dlib의 pre-trained 얼굴 감지 모델 사용)
detector = dlib.get_frontal_face_detector()

for path in paths:

    # PDF 파일 열기
    doc = fitz.open(path)

    # 이메일, 전화번호, 교육기관명 검색 및 추가
    for page in doc:
        text = page.get_text("text")

        email_addresses = re.findall(email_pattern, text)
        for email in email_addresses:
            if email not in search_texts:
                search_texts.append(email)
                replace_texts.append("[email]")

        phone_numbers = re.findall(phone_pattern, text)
        for phone in phone_numbers:
            if phone not in search_texts:
                search_texts.append(phone)
                replace_texts.append("[phone]")

        edus = re.findall(edu_pattern, text)
        for edu in edus:
            if edu not in search_texts:
                search_texts.append(edu)
                replace_texts.append("[edu]")

        urls = re.findall(url_pattern, text)
        for url in urls:
            if url not in search_texts:
                search_texts.append(url)
                replace_texts.append("[url]")

        url_texts.extend(urls)

        # 하이퍼링크 추출 및 대체
        for page in doc:
            for link in page.get_links():
                if 'uri' in link:
                    uri = link['uri']
                    # 모든 하이퍼링크를 '[link]'로 대체
                    search_texts.append(uri)
                    replace_texts.append('[link]')

            for link in page.get_links():
                if 'uri' in link:
                    uri = link['uri']
                    # URL을 리스트에 추가
                    url_texts.append(uri)

    # 임시 이미지 파일들을 담을 리스트
    temp_images = []

    # 페이지별로 작업
    for page_num, page in enumerate(doc):
        # 검색어 대체
        for search_text, replace_text in zip(search_texts, replace_texts):

            ### txt 마스킹 작업 ###
            if search_text in text:
                text = re.sub(re.escape(search_text), replace_text, text)
                substitution_counts[f"{search_text} -> {replace_text}"] += 1

            ### pdf 마스킹 작업 ###
            text_instances = page.search_for(search_text)
            for inst in text_instances:
                x0, y0, x1, y1 = inst

                # 기존 텍스트가 있던 영역을 흰색으로 덮어쓰기
                page.draw_rect(inst, color=(1, 1, 1), fill=(1, 1, 1))

                # 대체 텍스트 삽입
                page.insert_text((x0, (y0 + y1) // 2), replace_text, fontsize=8, fontname="helv")

                print(f"Page {page_num + 1} - '{search_text}' 위치: ({x0}, {y0}, {x1}, {y1})")

        ### txt 마스킹 작업 ###
        text = page.get_text()
        text = ' '.join(text.split())
        for search_text, replace_text in zip(search_texts, replace_texts):
            if search_text in text:
                text = re.sub(re.escape(search_text), replace_text, text)
                substitution_counts[f"{search_text} -> {replace_text}"] += 1
        masked_texts.append(f"Page {page_num + 1}:\n{text}")


        ### pdf 마스킹 작업 ###

        # 페이지를 이미지로 변환하여 임시 파일로 저장
        dpi = 300
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        temp_image_path = f"temp_page_{page_num + 1}.png"
        img.save(temp_image_path, quality=95) #여기서 이미지 저장
        temp_images.append(temp_image_path) #이미지 저장한 경로를 리스트에 넣음

        # 얼굴 감지 및 대체
        img_np = np.array(img)
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = detector(gray_img)

        # 텍스트와 원을 그릴 PIL 이미지 생성
        pil_img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil_img)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # 원 그리기 (외곽선 없음)
            draw.ellipse((x, y, x+w, y+h), fill="white", outline=None)

            # '[Face]' 텍스트 그리기
            text = "[Face]"
            #text_width, text_height = draw.textsize(text, font=font)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x = x + (w - text_width) / 2
            text_y = y + (h - text_height) / 2
            draw.text((text_x, text_y), text, fill="black", font=font)

        # 수정된 이미지를 저장
        replaced_img_path = temp_image_path  # 수정된 이미지를 같은 경로에 저장
        pil_img.save(replaced_img_path, quality=95) #여기서 또 저장

    # PDF를 이미지로 변환하여 저장된 이미지들을 새로운 PDF로 저장
    path_without_extension = os.path.splitext(path)[0] #파일 이름 지정을 위해서 만든것
    output_path = path_without_extension + "_마스킹.pdf"

    
    image_list = [Image.open(img_path) for img_path in temp_images]
    image_list[0].save(output_path, save_all=True, append_images=image_list[1:]) 

    # 임시 이미지 파일 삭제
    for temp_img in temp_images:
        os.remove(temp_img)

    # 문서 닫기
    doc.close()

    ### txt 마스킹 작업 ###
    print(masked_texts)
    output_path = f"{os.path.splitext(path)[0]}_마스킹.txt"
    with open(output_path, 'w', encoding='utf-8') as file:
        for masked_text in masked_texts:
            file.write(masked_text + "\n\n")
    if substitution_counts:
        print(f"\n| ---------- 대체된 항목 및 대체 횟수 ---------- |")
        for substitution, count in substitution_counts.items():
            print(f"{substitution} : {count}개")

    ### pdf 마스킹 작업 ###
    if "이력서" in path:
        print("\n| ----------        이력서 마스킹 완료        ---------- |")
    else:
        print("\n| ----------      포트폴리오 마스킹 완료      ---------- |")
    print(f"마스킹 된 파일이 다음 경로에 저장되었습니다.\n{output_path}\n")

    if url_texts:
        # 리스트를 집합으로 변환하여 중복 제거
        url_texts = list(set(url_texts))
        # URL을 텍스트 파일에 기록
        with open(url_log_path, 'w') as file:
            for url in url_texts:
                file.write(f"{url}\n")

        print(f"마스킹된 URL 목록이 다음 경로에 저장되었습니다.\n{url_log_path}")
