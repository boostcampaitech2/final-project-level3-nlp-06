import re
def remove_useless_breacket(text):
    """
    위키피디아 전처리를 위한 함수입니다.
    괄호 내부에 의미가 없는 정보를 제거합니다.
    아무런 정보를 포함하고 있지 않다면, 괄호를 통채로 제거합니다.
    ``수학(,)`` -> ``수학``
    ``수학(數學,) -> ``수학(數學)``
    """
    bracket_pattern = re.compile(r"\((.*?)\)")
    # preprocessed_text = []
    # for text in texts:
    modi_text = ""
    text = text.replace("()", "")  # 수학() -> 수학
    brackets = bracket_pattern.search(text)
    if not brackets:
        if text:
            return text
            # preprocessed_text.append(text)
            # continue
    replace_brackets = {}
    # key: 원본 문장에서 고쳐야하는 index, value: 고쳐져야 하는 값
    # e.g. {'2,8': '(數學)','34,37': ''}
    while brackets:
        index_key = str(brackets.start()) + "," + str(brackets.end())
        bracket = text[brackets.start() + 1 : brackets.end() - 1]
        infos = bracket.split(",")
        modi_infos = []
        for info in infos:
            info = info.strip()
            if len(info) > 0:
                modi_infos.append(info)
        if len(modi_infos) > 0:
            replace_brackets[index_key] = "(" + ", ".join(modi_infos) + ")"
        else:
            replace_brackets[index_key] = ""
        brackets = bracket_pattern.search(text, brackets.start() + 1)
    end_index = 0
    for index_key in replace_brackets.keys():
        start_index = int(index_key.split(",")[0])
        modi_text += text[end_index:start_index]
        modi_text += replace_brackets[index_key]
        end_index = int(index_key.split(",")[1])
    modi_text += text[end_index:]
    modi_text = modi_text.strip()
    if modi_text:
        return modi_text
            # preprocessed_text.append(modi_text)
    # return preprocessed_text

def clean_punc(text):
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

    # preprocessed_text = []
    # for text in texts:
    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])
    text = text.strip()
    if text:
        return text
            # preprocessed_text.append(text)
    # return preprocessed_text


def remove_repeated_spacing(text):
    """
    두 개 이상의 연속된 공백을 하나로 치환합니다.
    ``오늘은    날씨가   좋다.`` -> ``오늘은 날씨가 좋다.``
    """
    # preprocessed_text = []
    # for text in texts:
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        return text
            # preprocessed_text.append(text)
    # return preprocessed_text

def remove_email(text):
    """
    이메일을 제거합니다.
    ``홍길동 abc@gmail.com 연락주세요!`` -> ``홍길동  연락주세요!``
    """
    # preprocessed_text = []
    # for text in texts:
    text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text).strip()
    if text:
        return text
    #         preprocessed_text.append(text)
    # return preprocessed_text

def remove_url(text):
    """
    URL을 제거합니다.
    ``주소: www.naver.com`` -> ``주소: ``
    """
    # preprocessed_text = []
    # for text in texts:
    text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text).strip()
    text = re.sub(r"pic\.(\w+\.)+\S*", "", text).strip()
    if text:
        return text
            # preprocessed_text.append(text)
    # return preprocessed_text