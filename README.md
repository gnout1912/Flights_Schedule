# B1. Tải MySQL WORKBENCH, SHELL, SERVER (Cài đúng 3 cái đó rồi đặt mật khẩu).
# B2. Mở MySQL WORKBENCH. Copy cái đống command trong file source/sql.sql vô rồi bấm cái icon dấu sét màu vàng.
# B3. Bấm CTRL + Shift + P ==> Chọn Python: Select Interpreter rồi chọn 1 cái môi trường.
# B4. Mở Terminal (CTRL + Shift + `). Rồi nhập command: 
#   -cd model. 
#   -python train_model.py ==> Xong mở cái source coi nó có tạo ra 3 file .pkl chưa.
# B5. Trong Terminal. Nhập command: 
#   - cd ..
#   - pip install -r requirements.txt
#   - cd source 
# B6. Vô app.py, ở dòng 19 đổi thành cái password ban nãy đặt lúc tải ở B1.
# B7. Trong Terminal. Nhập command: python app.py ==> Thành công rồi thì lên web http://127.0.0.1:5000
# B8. Muốn test cái lịch nào thì bỏ file vô trong thư mục file_upload rồi up lên