import os
import json
import pandas as pd
import google.generativeai as genai
import logging

# Kiểm tra và tạo thư mục logs nếu chưa tồn tại
log_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Cấu hình Google Gemini API
genai.configure(api_key='AIzaSyCOrTaS_X_lgPTU12Y_Aul_2eV7VX_d3hk')

# Thiết lập logging
logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'toilet_analysis.log'),
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Danh sách các từ khóa cần loại bỏ
keywords_to_exclude = ["Tập thể", "Chung cư", "TT"]

# Hàm gọi Google Gemini AI để phân loại "Mặt phố" hoặc "Ngõ"
def classify_street_or_alley(model, description, street):
    """
    Gọi API Gemini để phân loại căn nhà là 'mặt phố' hay 'ngõ' dựa trên description và street.
    """
    prompt = f"Analyze the following real estate description and street information to determine if it is 'mặt phố' or 'ngõ'. If it's 'mặt ngõ' or 'ngõ', return 'ngõ'. If it's 'mặt phố', return 'mặt phố'. If both are mentioned, assume 'ngõ'. Description: {description}. Street: {street}"

    try:
        response = model.generate_content(prompt)
        # Lấy phản hồi từ API
        response_text = response.text.strip().lower()

        # Trả về giá trị tương ứng
        if 'ngõ' in response_text or 'mặt ngõ' in response_text:
            return 'ngõ'
        elif 'mặt phố' in response_text:
            return 'mặt phố'
        else:
            return 'không rõ'
    except Exception as e:
        logging.error(f"Error calling Gemini API for street classification: {e}")
        return 'không rõ'  # Nếu lỗi, trả về 'không rõ'


# Hàm gọi Google Gemini API để phân tích mô tả và trích xuất số lượng toilet và room
def extract_info_with_gemini(model, description):
    """
    Gọi API Gemini để trích xuất số lượng toilet và room từ mô tả.
    """
    prompt = f"Analyze the following real estate description and return the number of toilets and rooms as integer values in JSON format. If no toilet or room is mentioned, return 0. Description: {description}"

    try:
        response = model.generate_content(prompt)
        data = json.loads(response.text)  # Giả sử phản hồi là JSON có toilet_count và room_count
        toilet_count = data.get('toilet_count', 0)
        room_count = data.get('room_count', 0)
        return toilet_count, room_count
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return 0, 0  # Nếu lỗi, trả về 0 toilets và 0 rooms


def clean_street_and_attributes(model_street, description, street, attributes):
    """
    Sử dụng model Gemini AI để phân loại 'mặt phố' hay 'ngõ', và cập nhật thông tin vào attributes.
    """
    # Gọi model Gemini để phân loại loại đường (mặt phố, ngõ, không rõ)
    street_type = classify_street_or_alley(model_street, description, street)

    # Xử lý thông tin dựa trên street_type
    if street_type == 'ngõ':
        # Xóa "mặt phố" khỏi attributes nếu có
        if "mặt phố" in attributes:
            attributes.remove("mặt phố")
        # Xóa từ "ngõ" hoặc "mặt ngõ" khỏi street và thêm "ngõ" vào attributes nếu chưa có
        street = street.replace("ngõ", "").replace("mặt ngõ", "").strip()
        if "ngõ" not in attributes:
            attributes.append("ngõ")
    elif street_type == 'mặt phố':
        # Xóa từ "mặt phố" khỏi street và thêm "mặt phố" vào attributes nếu chưa có
        street = street.replace("mặt phố", "").strip()
        if "mặt phố" not in attributes:
            attributes.append("mặt phố")

    return street, attributes


def load_json_files_to_matrix(directory):
    """
    Đọc tất cả các tệp JSON trong thư mục, trích xuất thông tin và trả về DataFrame.
    """
    data = []

    # Tạo đối tượng model1 từ Google Gemini để phân loại 'mặt phố' hoặc 'ngõ'
    model_street = genai.GenerativeModel(
        "gemini-1.5-flash-001",
        generation_config={"response_mime_type": "application/json", "max_output_tokens": 8192, "temperature": 0,
                           "top_p": 0.7},
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        system_instruction="""
    I'm a developer in a real estate platform. 
    Please help me classify real estate listings into 'mặt phố' or 'ngõ' based on the description and street information provided. 
    Use the following rules:

    1. If the description or street contains the term 'mặt phố', it should be classified as 'mặt phố'.
    2. If the description or street contains the term 'ngõ' or 'mặt ngõ', it should be classified as 'ngõ'.
    3. If both 'mặt phố' and 'ngõ' (or 'mặt ngõ') are present in either the description or street, always classify as 'ngõ'.
    4. After classification:
        - For 'ngõ': Remove the terms 'ngõ' or 'mặt ngõ' from the street information and add 'ngõ' to the attributes. If 'mặt phố' exists in the attributes, remove it.
        - For 'mặt phố': Remove the term 'mặt phố' from the street information and add 'mặt phố' to the attributes, but only if it is not already present.
    5. If none of the above terms are found, classify as 'không rõ'.
    6. Output should be one of the following strings: 'mặt phố', 'ngõ', or 'không rõ'.
    """
    )

    # Tạo đối tượng model2 từ Google Gemini để trích xuất số lượng toilet và room
    model_info = genai.GenerativeModel(
        "gemini-1.5-flash-001",
        generation_config={"response_mime_type": "application/json", "max_output_tokens": 8192, "temperature": 0,
                           "top_p": 0.7},
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        system_instruction="""
        I'm a developer in a real estate platform. 
        Please help me extract the number of toilets and rooms from the description as integer values in JSON format. 
        If no toilet or room is mentioned, return 0 for each. 
        Use the following rules:
        1. Recognize all variations of the term 'toilet', including but not limited to: 'vệ sinh', 'toilet', 'nhà vệ sinh', 'VS', 'WC', 'nhà tắm', 'phòng tắm', and other possible synonyms or abbreviations.
        2. Recognize all variations of the term 'room', including but not limited to: 'phòng', 'phòng ngủ', 'PN', 'chỗ ở', and other similar terms.
        3. If only the number of rooms is mentioned, assume the number of toilets equals the number of rooms.
        4. Ignore irrelevant information and focus only on extracting the toilet and room counts from the text.
        5. Output should always be in JSON format with fields 'toilet_count' and 'room_count'.
    """
    )

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            logging.info(f"Đang xử lý tệp: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    file_content = file.read().strip()
                    if file_content:
                        file_data = json.loads(file_content)

                        for house in file_data:
                            description = house.get("description", "").strip()
                            attributes = house.get("attribute", [])

                            # Loại bỏ căn nhà nếu description hoặc attribute chứa từ khóa không mong muốn
                            if any(keyword.lower() in description.lower() for keyword in keywords_to_exclude) or \
                               any(keyword.lower() in attribute.lower() for attribute in attributes for keyword in keywords_to_exclude):
                                logging.info(f"Bỏ qua căn nhà vì chứa từ khóa loại bỏ: {description} hoặc {attributes}")
                                continue

                            # Gọi model để trích xuất số lượng toilet và room
                            toilet_count, room_count = extract_info_with_gemini(model_info, description)

                            # Lấy giá trị street
                            street = house["location"]["street"]

                            # Sử dụng model để làm sạch street và attributes dựa trên yêu cầu
                            street, attributes = clean_street_and_attributes(model_street, description, street, attributes)

                            # Lấy số tầng và cập nhật nếu bằng 0
                            floor = house["additional"].get("floor", 0)
                            if floor == 0:
                                floor = 1

                            # Thêm thông tin căn nhà vào danh sách
                            data.append({
                                "province": house["location"]["province"],
                                "district": house["location"]["district"],
                                "street": street,  # Đã cập nhật thông tin street
                                "price_unit": house["price"]["unit"],
                                "price_value": house["price"]["value"],
                                "price_absolute": house["price"]["absolute"],
                                "area": house["area"],
                                "type": house["type"],
                                "front": house["additional"]["front"],
                                "room": room_count,
                                "toilet": toilet_count,
                                "floor": floor,
                                "attributes": ", ".join(attributes),  # Đã cập nhật thông tin attributes
                            })
            except json.JSONDecodeError:
                logging.error(f"File {filename} không phải là tệp JSON hợp lệ hoặc rỗng.")
            except KeyError as e:
                logging.error(f"Lỗi thiếu dữ liệu {e} trong tệp {filename}")

    df = pd.DataFrame(data)
    return df


# Gọi hàm để tải dữ liệu từ tất cả các file JSON trong thư mục hiện tại
data_dir = os.getcwd()
df = load_json_files_to_matrix(data_dir)

# Lọc bỏ các dòng có giá trị 'front' bằng 1
df = df[df['front'] != 1]

# Đặt các tùy chọn hiển thị để hiển thị toàn bộ DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# In toàn bộ DataFrame
print(df)

# Lưu DataFrame vào tệp CSV và Excel
output_csv_path = os.path.join(data_dir, 'dongda.csv')
output_excel_path = os.path.join(data_dir, 'dongda.xlsx')

df.to_csv(output_csv_path, index=False)
df.to_excel(output_excel_path, index=False)

logging.info(f"Đã lưu file CSV tại: {output_csv_path}")
logging.info(f"Đã lưu file Excel tại: {output_excel_path}")
