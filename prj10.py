import numpy as np
import cv2
import os
import glob
import json

sbd_goc = [24, 33]
toa_do_sbd = [[sbd_goc[0] + i * 47, sbd_goc[1] + j * 66] for i in range(6) for j in range(10)]

mdt_goc = [24, 33]
toa_do_mdt = [[mdt_goc[0] + i * 47, mdt_goc[1] + j * 66] for i in range(3) for j in range(10)]

p1_goc = [118, 94]
toa_do_p1 = [[p1_goc[0] + j * 94, p1_goc[1] + i * 56] for i in range(10) for j in range(4)]

p2_goc = [117, 157]
toa_do_p2 = [[p2_goc[0] + i * 188 + k * 94, p2_goc[1] + j * 54] for i in range(2) for j in range(4) for k in range(2)]

p3_goc = [110, 180]
toa_do_p3 = [[p3_goc[0] + 342 * x + i * 58, p3_goc[1] + j * 57] for x in range(6) for i in range(4) for j in range(12)]


def kiem_tra_to_den(vung_anh, nguong_dien=0.3, trung_binh=100):
    if len(vung_anh.shape) == 3:
        vung_anh = cv2.cvtColor(vung_anh, cv2.COLOR_BGR2GRAY)
    _, anh_nhi_phan = cv2.threshold(vung_anh, trung_binh, 255, cv2.THRESH_BINARY_INV)
    tong_diem_anh = anh_nhi_phan.size
    anh_an_mon = cv2.erode(anh_nhi_phan, np.ones((3, 3), np.uint8), iterations=1)
    diem_anh_den = cv2.countNonZero(anh_an_mon)
    ty_le_dien = diem_anh_den / tong_diem_anh
    return ty_le_dien > nguong_dien


def tim_nguong(anh, trung_binh=100, so_lan_lap=1):
    if len(anh.shape) == 3:
        anh = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    gia_tri = 0
    for j in range(80, 0, -1):
        _, anh_nhi_phan = cv2.threshold(anh, trung_binh + j, 255, cv2.THRESH_BINARY_INV)
        anh_an_mon = cv2.erode(anh_nhi_phan, np.ones((3, 3), np.uint8), iterations=so_lan_lap)
        diem_anh_den = cv2.countNonZero(anh_an_mon)
        tong_diem_anh = anh_an_mon.size
        ty_le_dien = diem_anh_den / tong_diem_anh
        if ty_le_dien < 0.12:
            gia_tri = j
            break
    return gia_tri


def tien_xu_ly_anh(duong_dan_anh):
    anh_goc = cv2.imread(duong_dan_anh, 0)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(4, 4))
    anh_can_bang = clahe.apply(anh_goc)
    anh_lam_mo = cv2.fastNlMeansDenoising(anh_can_bang, h=50, templateWindowSize=7, searchWindowSize=21)
    anh_nhi_phan = cv2.adaptiveThreshold(anh_lam_mo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 3)
    return anh_nhi_phan, anh_goc


def cat_cac_phan(anh_nhi_phan, anh_goc):
    y, x = anh_nhi_phan.shape
    thong_tin = anh_nhi_phan[:int(y / 2.5), int(x / 1.5):]
    phan_1 = anh_nhi_phan[int(y / 3.3):int(y / 1.6)]
    phan_2 = anh_nhi_phan[int(y / 2):int(y / 1.3)]
    phan_3 = anh_nhi_phan[int(y / 1.6):]

    anh_thong_tin = anh_goc[:int(y / 2.5), int(x / 1.5):]
    anh_phan_1 = anh_goc[int(y / 3.3):int(y / 1.6)]
    anh_phan_2 = anh_goc[int(y / 2):int(y / 1.3)]
    anh_phan_3 = anh_goc[int(y / 1.6):]

    return thong_tin, phan_1, phan_2, phan_3, anh_thong_tin, anh_phan_1, anh_phan_2, anh_phan_3


def sap_xep_toa_do(toa_do):
    hinh_chu_nhat = np.zeros((4, 2), dtype="float32")
    tong = toa_do.sum(axis=1)
    hieu = np.diff(toa_do, axis=1)
    hinh_chu_nhat[0] = toa_do[np.argmin(tong)]
    hinh_chu_nhat[2] = toa_do[np.argmax(tong)]
    hinh_chu_nhat[1] = toa_do[np.argmin(hieu)]
    hinh_chu_nhat[3] = toa_do[np.argmax(hieu)]
    return hinh_chu_nhat


def lay_tu_giac(vien):
    chu_vi = cv2.arcLength(vien, True)
    da_giac = cv2.approxPolyDP(vien, 0.02 * chu_vi, True)
    if len(da_giac) == 4:
        tu_giac = da_giac.reshape(4, 2)
        return sap_xep_toa_do(tu_giac)
    else:
        x, y, rong, cao = cv2.boundingRect(vien)
        tu_giac = np.array([[x, y], [x + rong, y], [x + rong, y + cao], [x, y + cao]])
        return sap_xep_toa_do(tu_giac)


def xoay_phang_anh(anh, toa_do, rong, cao):
    toa_do_dich = np.array([[0, 0], [rong - 1, 0], [rong - 1, cao - 1], [0, cao - 1]], dtype="float32")
    ma_tran = cv2.getPerspectiveTransform(toa_do, toa_do_dich)
    ma_tran_nguoc = cv2.getPerspectiveTransform(toa_do_dich, toa_do)
    anh_da_xoay = cv2.warpPerspective(anh, ma_tran, (rong, cao))
    return anh_da_xoay, ma_tran_nguoc


def khop_mau_homography():
    anh_mau = cv2.imread('THPT2025.png', 0)
    anh_quet = cv2.imread(duong_dan_phieu, 0)
    le_anh = 300
    mau_trang = [255, 255, 255]
    anh_quet_vien = cv2.copyMakeBorder(anh_quet, le_anh, le_anh, le_anh, le_anh, cv2.BORDER_CONSTANT, value=mau_trang)

    akaze = cv2.AKAZE_create()
    diem_mau, dac_trung_mau = akaze.detectAndCompute(anh_mau, None)
    diem_quet, dac_trung_quet = akaze.detectAndCompute(anh_quet_vien, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(dac_trung_mau, dac_trung_quet, k=2)

    diem_khop_tot = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            diem_khop_tot.append(m)

    toa_do_nguon = np.float32([diem_mau[m.queryIdx].pt for m in diem_khop_tot]).reshape(-1, 1, 2)
    toa_do_dich = np.float32([diem_quet[m.trainIdx].pt for m in diem_khop_tot]).reshape(-1, 1, 2)

    ma_tran_h, _ = cv2.findHomography(toa_do_dich, toa_do_nguon, cv2.RANSAC, 5.0)
    ma_tran_h_nguoc = np.linalg.inv(ma_tran_h)

    cao, rong = anh_mau.shape
    anh_da_chinh = cv2.warpPerspective(anh_quet_vien, ma_tran_h, (rong, cao))

    mang_thong_tin = [anh_da_chinh[425:1085, 1775:2050], anh_da_chinh[425:1085, 2130:2270]]
    mang_p1 = [anh_da_chinh[1220:1855, 200 + 530 * i:645 + 530 * i] for i in range(4)]
    mang_p2 = [anh_da_chinh[1945:2315, 200 + 530 * i:645 + 530 * i] for i in range(4)]
    mang_p3 = [anh_da_chinh[2440:3250, 200:2235]]

    return mang_thong_tin, mang_p1, mang_p2, mang_p3, ma_tran_h_nguoc


def ve_ket_qua_len_anh_goc(toa_do, offset_x, offset_y, is_homo, m_inv, color=(0, 255, 0)):
    pt = np.array([[[toa_do[0], toa_do[1]]]], dtype="float32")
    if is_homo:
        pt[0][0][0] += offset_x
        pt[0][0][1] += offset_y
        pt_vien = cv2.perspectiveTransform(pt, m_inv)
        x = int(pt_vien[0][0][0]) - 300
        y = int(pt_vien[0][0][1]) - 300
    else:
        pt_phan = cv2.perspectiveTransform(pt, m_inv)
        x = int(pt_phan[0][0][0]) + offset_x
        y = int(pt_phan[0][0][1]) + offset_y

    cv2.circle(anh_xuat, (x, y), radius=8, color=color, thickness=3)


def tim_cac_vien(anh_nhi_phan):
    cao, rong = anh_nhi_phan.shape[:2]
    dien_tich = cao * rong
    danh_sach_vien = []
    cac_vien, _ = cv2.findContours(anh_nhi_phan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for v in cac_vien:
        if cv2.contourArea(v) > dien_tich / 20:
            danh_sach_vien.append(v)
    return sorted(danh_sach_vien, key=lambda x: cv2.boundingRect(x)[0])


def xu_ly_thong_tin(anh_nhi_phan, anh_goc_phan, off_x, off_y):
    rong, cao = 280, 660
    cac_vien = tim_cac_vien(anh_nhi_phan)
    is_homo = False
    if len(cac_vien) != 2:
        global ket_qua_homography
        if 'ket_qua_homography' not in globals() or ket_qua_homography is None:
            ket_qua_homography = khop_mau_homography()
        cac_vien = ket_qua_homography[0]
        khoi_1, khoi_2 = cac_vien[0], cac_vien[1]
        m_inv = ket_qua_homography[4]
        is_homo = True
    else:
        toa_do_1 = lay_tu_giac(cac_vien[0])
        khoi_1, m_inv_1 = xoay_phang_anh(anh_goc_phan, toa_do_1, rong, cao)
        toa_do_2 = lay_tu_giac(cac_vien[1])
        khoi_2, m_inv_2 = xoay_phang_anh(anh_goc_phan, toa_do_2, int(rong / 2), cao)

    do_lech_chuan = np.std(khoi_1)
    trung_binh = np.mean(khoi_1) - do_lech_chuan
    trung_binh += tim_nguong(khoi_1, trung_binh)

    so_bao_danh = ''
    sbd_positions = [False] * 6  # Theo dõi vị trí đã tô cho SBD (6 ô)
    diem_ve_sbd_cuoi = []
    for i in range(8, 0, -1):
        sbd_digits = [''] * 6
        nguong_dien = 0.1 * i
        diem_tam = []
        sbd_temp = [False] * 6
        for vi_tri, toa_do in enumerate(toa_do_sbd):
            vung_o = khoi_1[toa_do[1] - 18:toa_do[1] + 18, toa_do[0] - 18:toa_do[0] + 18]
            if kiem_tra_to_den(vung_o, nguong_dien, trung_binh):
                col = vi_tri // 10
                if sbd_digits[col] == '':
                    sbd_digits[col] = str(vi_tri % 10)
                    sbd_temp[col] = True
                    diem_tam.append(toa_do)

        filled_count = sum(1 for d in sbd_digits if d != '')
        if filled_count == 6:
            so_bao_danh = ''.join(sbd_digits)
            sbd_positions = sbd_temp
            diem_ve_sbd_cuoi = diem_tam
            break
        elif 0 < filled_count < 6:
            so_bao_danh = ''.join(d for d in sbd_digits if d != '')
            sbd_positions = sbd_temp
            diem_ve_sbd_cuoi = diem_tam

    for toa_do in diem_ve_sbd_cuoi:
        if is_homo:
            ve_ket_qua_len_anh_goc(toa_do, 1775, 425, True, m_inv, (0, 255, 0))
        else:
            ve_ket_qua_len_anh_goc(toa_do, off_x, off_y, False, m_inv_1, (0, 255, 0))

    do_lech_chuan_2 = np.std(khoi_2)
    trung_binh_2 = np.mean(khoi_2) - do_lech_chuan_2
    trung_binh_2 += tim_nguong(khoi_2, trung_binh_2)

    ma_de_thi = ''
    mdt_positions = [False] * 3
    diem_ve_mdt_cuoi = []
    for i in range(8, 0, -1):
        mdt_digits = [''] * 3
        nguong_dien = 0.1 * i
        diem_tam = []
        mdt_temp = [False] * 3
        for vi_tri, toa_do in enumerate(toa_do_mdt):
            vung_o = khoi_2[toa_do[1] - 18:toa_do[1] + 18, toa_do[0] - 18:toa_do[0] + 18]
            if kiem_tra_to_den(vung_o, nguong_dien, trung_binh_2):
                col = vi_tri // 10
                if mdt_digits[col] == '':
                    mdt_digits[col] = str(vi_tri % 10)
                    mdt_temp[col] = True
                    diem_tam.append(toa_do)

        filled_count = sum(1 for d in mdt_digits if d != '')
        if filled_count == 3:
            ma_de_thi = ''.join(mdt_digits)
            mdt_positions = mdt_temp
            diem_ve_mdt_cuoi = diem_tam
            break
        elif 0 < filled_count < 3:
            ma_de_thi = ''.join(d for d in mdt_digits if d != '')
            mdt_positions = mdt_temp
            diem_ve_mdt_cuoi = diem_tam

    for toa_do in diem_ve_mdt_cuoi:
        if is_homo:
            ve_ket_qua_len_anh_goc(toa_do, 2130, 425, True, m_inv, (0, 255, 0))
        else:
            ve_ket_qua_len_anh_goc(toa_do, off_x, off_y, False, m_inv_2, (0, 255, 0))

    return so_bao_danh, ma_de_thi, sbd_positions, mdt_positions


def xu_ly_phan_1(anh_nhi_phan, anh_goc_phan, off_x, off_y):
    global info_phan_1
    
    cac_vien = tim_cac_vien(anh_nhi_phan)
    is_homo = False
    cac_khoi, cac_minv = [], []
    if len(cac_vien) != 4:
        global ket_qua_homography
        if 'ket_qua_homography' not in globals() or ket_qua_homography is None:
            ket_qua_homography = khop_mau_homography()
        cac_khoi = ket_qua_homography[1]
        m_inv = ket_qua_homography[4]
        is_homo = True
    else:
        for v in cac_vien:
            toa_do = lay_tu_giac(v)
            khoi, m_inv_k = xoay_phang_anh(anh_goc_phan, toa_do, 460, 640)
            cac_khoi.append(khoi)
            cac_minv.append(m_inv_k)
        m_inv = cac_minv[0] if cac_minv else None

    info_phan_1 = {
        'is_homo': is_homo,
        'm_inv': m_inv,
        'cac_minv': cac_minv,
        'off_x': 200 if is_homo else off_x,
        'off_y': 1220 if is_homo else off_y
    }

    dap_an_p1 = [[] for _ in range(40)]
    for i, khoi in enumerate(cac_khoi):
        do_lech = np.std(khoi)
        trung_binh = np.mean(khoi) - do_lech
        gia_tri_nguong = tim_nguong(khoi, trung_binh)
        trung_binh = trung_binh + gia_tri_nguong

        for vi_tri, toa_do in enumerate(toa_do_p1):
            vung_o = khoi[toa_do[1] - 18:toa_do[1] + 18, toa_do[0] - 18:toa_do[0] + 18]
            if kiem_tra_to_den(vung_o, 0.4, trung_binh):
                so_canh_trong_cau = len(dap_an_p1[(vi_tri // 4) + 10 * i])
                color = (0, 0, 255) if so_canh_trong_cau > 0 else (0, 255, 0)  # Đỏ nếu đã có, xanh nếu lần đầu
                
                if is_homo:
                    ve_ket_qua_len_anh_goc(toa_do, 200 + 530 * i, 1220, True, m_inv, color)
                else:
                    ve_ket_qua_len_anh_goc(toa_do, off_x, off_y, False, cac_minv[i], color)
                dap_an_p1[(vi_tri // 4) + 10 * i].append(vi_tri % 4)

    return dap_an_p1


def xu_ly_phan_2(anh_nhi_phan, anh_goc_phan, off_x, off_y):
    global info_phan_2
    
    cac_vien = tim_cac_vien(anh_nhi_phan)
    is_homo = False
    cac_khoi, cac_minv, cac_off_x, cac_off_y = [], [], [], []
    if len(cac_vien) != 4:
        global ket_qua_homography
        if 'ket_qua_homography' not in globals() or ket_qua_homography is None:
            ket_qua_homography = khop_mau_homography()
        cac_khoi = ket_qua_homography[2]
        m_inv = ket_qua_homography[4]
        is_homo = True
    else:
        for v in cac_vien:
            x, y, w, h = cv2.boundingRect(v)
            toa_do = lay_tu_giac(v)
            khoi, m_inv_k = xoay_phang_anh(anh_goc_phan, toa_do, 460, 360)
            cac_khoi.append(khoi)
            cac_minv.append(m_inv_k)
            cac_off_x.append(x)
            cac_off_y.append(y)
        m_inv = cac_minv[0] if cac_minv else None

    info_phan_2 = {
        'is_homo': is_homo,
        'm_inv': m_inv,
        'cac_minv': cac_minv,
        'cac_off_x': cac_off_x,
        'cac_off_y': cac_off_y,
        'off_x': 200 if is_homo else off_x,
        'off_y': 1945 if is_homo else off_y
    }

    dap_an_p2 = [[] for _ in range(32)]
    for i, khoi in enumerate(cac_khoi):
        do_lech = np.std(khoi)
        trung_binh = np.mean(khoi) - do_lech
        trung_binh += tim_nguong(khoi, trung_binh)

        for vi_tri, toa_do in enumerate(toa_do_p2):
            vung_o = khoi[toa_do[1] - 18:toa_do[1] + 18, toa_do[0] - 18:toa_do[0] + 18]
            if kiem_tra_to_den(vung_o, 0.3, trung_binh):
                so_canh_trong_cau = len(dap_an_p2[(vi_tri // 2) + 8 * i])
                color = (0, 0, 255) if so_canh_trong_cau > 0 else (0, 255, 0)
                
                if is_homo:
                    ve_ket_qua_len_anh_goc(toa_do, 200 + 530 * i, 1945, True, m_inv, color)
                else:
                    ve_ket_qua_len_anh_goc(toa_do, off_x, off_y, False, cac_minv[i], color)
                dap_an_p2[(vi_tri // 2) + 8 * i].append((vi_tri + 1) % 2)

    return dap_an_p2


def xu_ly_phan_3(anh_nhi_phan, anh_goc_phan, off_x, off_y):
    global info_phan_3
    
    cac_vien = tim_cac_vien(anh_nhi_phan)
    is_homo = False
    cac_khoi, cac_minv = [], []
    if len(cac_vien) != 1:
        global ket_qua_homography
        if 'ket_qua_homography' not in globals() or ket_qua_homography is None:
            ket_qua_homography = khop_mau_homography()
        cac_khoi = ket_qua_homography[3]
        m_inv = ket_qua_homography[4]
        is_homo = True
    else:
        for v in cac_vien:
            toa_do = lay_tu_giac(v)
            khoi, m_inv_k = xoay_phang_anh(anh_goc_phan, toa_do, 2060, 860)
            cac_khoi.append(khoi)
            cac_minv.append(m_inv_k)
        m_inv = cac_minv[0] if cac_minv else None

    info_phan_3 = {
        'is_homo': is_homo,
        'm_inv': m_inv,
        'off_x': 200 if is_homo else off_x,
        'off_y': 2440 if is_homo else off_y
    }

    dap_an_p3 = [{'value': '', 'valid': True, 'positions': []} for _ in range(6)]
    cac_o_bo_qua = [1, 12, 24, 36, 37, 49, 60, 72, 84, 85, 97, 108, 120, 132, 133, 145, 156, 168, 180, 181, 193, 204,
                    216, 228, 229, 241, 252, 264, 276, 277]

    for idx_khoi, khoi in enumerate(cac_khoi):
        do_lech = np.std(khoi)
        trung_binh = np.mean(khoi) - do_lech
        trung_binh += tim_nguong(khoi, trung_binh)

        for q in range(6):
            cau_hople = True
            gia_tri_cau = ""
            for c in range(4):
                so_to_trong_cot = 0
                ky_tu_cot = ""

                for r in range(12):
                    vi_tri = q * 48 + c * 12 + r
                    if vi_tri in cac_o_bo_qua:
                        continue

                    toa_do = toa_do_p3[vi_tri]

                    y_start = max(0, toa_do[1] - 18)
                    y_end = min(khoi.shape[0], toa_do[1] + 18)
                    x_start = max(0, toa_do[0] - 18)
                    x_end = min(khoi.shape[1], toa_do[0] + 18)
                    vung_o = khoi[y_start:y_end, x_start:x_end]

                    if vung_o.size == 0:
                        continue

                    if kiem_tra_to_den(vung_o, 0.4, trung_binh):
                        color = (0, 0, 255) if so_to_trong_cot > 0 else (0, 255, 0)
                        
                        if is_homo:
                            ve_ket_qua_len_anh_goc(toa_do, 200, 2440, True, m_inv, color)
                        else:
                            ve_ket_qua_len_anh_goc(toa_do, off_x, off_y, False, cac_minv[idx_khoi], color)

                        so_to_trong_cot += 1
                        if r == 0:
                            ky_tu_cot = "-"
                        elif r == 1:
                            ky_tu_cot = ","
                        else:
                            ky_tu_cot = str(r - 2)

                if so_to_trong_cot > 1:
                    cau_hople = False
                elif so_to_trong_cot == 1:
                    gia_tri_cau += ky_tu_cot

            dap_an_p3[q]['value'] = gia_tri_cau
            dap_an_p3[q]['valid'] = cau_hople

    return dap_an_p3


def ve_gach_ngang_phan_3(q, is_homo, m_inv, off_x, off_y):
    vi_tri_dau = q * 48
    if vi_tri_dau >= len(toa_do_p3):
        return
    
    toa_do_start = toa_do_p3[vi_tri_dau]
    vi_tri_cuoi = min(q * 48 + 47, len(toa_do_p3) - 1)
    toa_do_end = toa_do_p3[vi_tri_cuoi]
    
    pt_start = np.array([[[toa_do_start[0], toa_do_start[1]]]], dtype="float32")
    pt_end = np.array([[[toa_do_end[0], toa_do_end[1]]]], dtype="float32")
    
    if is_homo:
        pt_start[0][0][0] += off_x
        pt_start[0][0][1] += off_y
        pt_end[0][0][0] += off_x
        pt_end[0][0][1] += off_y
        
        pt_start_homo = cv2.perspectiveTransform(pt_start, m_inv)
        pt_end_homo = cv2.perspectiveTransform(pt_end, m_inv)
        
        x_start = int(pt_start_homo[0][0][0]) - 300
        y_start = int(pt_start_homo[0][0][1]) - 300
        x_end = int(pt_end_homo[0][0][0]) - 300
        y_end = int(pt_end_homo[0][0][1]) - 300
    else:
        pt_start_phan = cv2.perspectiveTransform(pt_start, m_inv)
        pt_end_phan = cv2.perspectiveTransform(pt_end, m_inv)
        
        x_start = int(pt_start_phan[0][0][0]) + off_x
        y_start = int(pt_start_phan[0][0][1]) + off_y
        x_end = int(pt_end_phan[0][0][0]) + off_x
        y_end = int(pt_end_phan[0][0][1]) + off_y
    
    cv2.line(anh_xuat, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)


def ve_gach_ngang_phan_1(q, is_homo, m_inv, off_x, off_y):
    vi_tri_dau = q * 4
    if vi_tri_dau >= len(toa_do_p1):
        return
    
    toa_do_start = toa_do_p1[vi_tri_dau]
    vi_tri_cuoi = min(q * 4 + 3, len(toa_do_p1) - 1)
    toa_do_end = toa_do_p1[vi_tri_cuoi]
    
    pt_start = np.array([[[toa_do_start[0], toa_do_start[1]]]], dtype="float32")
    pt_end = np.array([[[toa_do_end[0], toa_do_end[1]]]], dtype="float32")
    
    if is_homo:
        pt_start[0][0][0] += off_x
        pt_start[0][0][1] += off_y
        pt_end[0][0][0] += off_x
        pt_end[0][0][1] += off_y
        
        pt_start_homo = cv2.perspectiveTransform(pt_start, m_inv)
        pt_end_homo = cv2.perspectiveTransform(pt_end, m_inv)
        
        x_start = int(pt_start_homo[0][0][0]) - 300
        y_start = int(pt_start_homo[0][0][1]) - 300
        x_end = int(pt_end_homo[0][0][0]) - 300
        y_end = int(pt_end_homo[0][0][1]) - 300
    else:
        pt_start_phan = cv2.perspectiveTransform(pt_start, m_inv)
        pt_end_phan = cv2.perspectiveTransform(pt_end, m_inv)
        
        x_start = int(pt_start_phan[0][0][0]) + off_x
        y_start = int(pt_start_phan[0][0][1]) + off_y
        x_end = int(pt_end_phan[0][0][0]) + off_x
        y_end = int(pt_end_phan[0][0][1]) + off_y
    
    cv2.line(anh_xuat, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)


def ve_gach_ngang_phan_2(q, is_homo, m_inv, off_x, off_y):
    vi_tri_dau = q * 2
    if vi_tri_dau >= len(toa_do_p2):
        return
    
    toa_do_start = toa_do_p2[vi_tri_dau]
    vi_tri_cuoi = min(q * 2 + 1, len(toa_do_p2) - 1)
    toa_do_end = toa_do_p2[vi_tri_cuoi]
    
    pt_start = np.array([[[toa_do_start[0], toa_do_start[1]]]], dtype="float32")
    pt_end = np.array([[[toa_do_end[0], toa_do_end[1]]]], dtype="float32")
    
    if is_homo:
        pt_start[0][0][0] += off_x
        pt_start[0][0][1] += off_y
        pt_end[0][0][0] += off_x
        pt_end[0][0][1] += off_y
        
        pt_start_homo = cv2.perspectiveTransform(pt_start, m_inv)
        pt_end_homo = cv2.perspectiveTransform(pt_end, m_inv)
        
        x_start = int(pt_start_homo[0][0][0]) - 300
        y_start = int(pt_start_homo[0][0][1]) - 300
        x_end = int(pt_end_homo[0][0][0]) - 300
        y_end = int(pt_end_homo[0][0][1]) - 300
    else:
        pt_start_phan = cv2.perspectiveTransform(pt_start, m_inv)
        pt_end_phan = cv2.perspectiveTransform(pt_end, m_inv)
        
        x_start = int(pt_start_phan[0][0][0]) + off_x
        y_start = int(pt_start_phan[0][0][1]) + off_y
        x_end = int(pt_end_phan[0][0][0]) + off_x
        y_end = int(pt_end_phan[0][0][1]) + off_y
    
    cv2.line(anh_xuat, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)


def hien_thi_sbd_mdt(sbd, mdt, sbd_positions=None, mdt_positions=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    sbd_display = ""
    if sbd_positions:
        sbd_list = list(sbd) if sbd else []
        index = 0
        for i in range(6):
            if sbd_positions[i]:
                sbd_display += sbd_list[index] if index < len(sbd_list) else "?"
                index += 1
            else:
                sbd_display += "-"
    else:
        sbd_display = sbd if sbd else "------"
    
    mdt_display = ""
    if mdt_positions:
        mdt_list = list(mdt) if mdt else []
        index = 0
        for i in range(3):
            if mdt_positions[i]:
                mdt_display += mdt_list[index] if index < len(mdt_list) else "?"
                index += 1
            else:
                mdt_display += "-"
    else:
        mdt_display = mdt if mdt else "---"
    
    sbd_mdt_data = {
        "sbd": sbd,
        "mdt": mdt,
        "sbd_display": sbd_display,
        "mdt_display": mdt_display
    }
    os.makedirs('Json', exist_ok=True)
    with open(os.path.join('Json', 'sbd_mdt.json'), 'w', encoding='utf-8') as f:
        json.dump(sbd_mdt_data, f, indent=2, ensure_ascii=False)
    
    y_text = 50
    x_text = anh_xuat.shape[1] - 350
    
    cv2.putText(anh_xuat, f"SBD: {sbd_display}", (x_text, y_text), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(anh_xuat, f"MDT: {mdt_display}", (x_text, y_text + 30), font, font_scale, (0, 0, 0), thickness)


def xuat_sbd_mdt_console(sbd, mdt, sbd_positions=None, mdt_positions=None):
    sbd_display = ""
    if sbd_positions:
        sbd_list = list(sbd) if sbd else []
        index = 0
        for i in range(6):
            if sbd_positions[i]:
                sbd_display += sbd_list[index] if index < len(sbd_list) else "?"
                index += 1
            else:
                sbd_display += "-"
    else:
        sbd_display = sbd if sbd else "------"
    
    mdt_display = ""
    if mdt_positions:
        mdt_list = list(mdt) if mdt else []
        index = 0
        for i in range(3):
            if mdt_positions[i]:
                mdt_display += mdt_list[index] if index < len(mdt_list) else "?"
                index += 1
            else:
                mdt_display += "-"
    else:
        mdt_display = mdt if mdt else "---"
    
    print(f"SBD: {sbd_display}")
    print(f"MDT: {mdt_display}")


def cham_phieu(file_phieu):
    global duong_dan_phieu
    duong_dan_phieu = file_phieu

    global anh_xuat
    anh_xuat = cv2.imread(file_phieu)

    ten_file = os.path.basename(file_phieu)
    ten_json = os.path.splitext(ten_file)[0] + '.json'
    duong_dan_goc = os.path.join('PhieuQG', ten_file)
    duong_dan_xuat = os.path.join('Output', ten_file)
    duong_dan_json = os.path.join('Json', ten_json)

    if 'ket_qua_homography' in globals():
        del globals()['ket_qua_homography']

    anh_nhi_phan, anh_goc = tien_xu_ly_anh(file_phieu)
    

    vung_thong_tin, vung_p1, vung_p2, vung_p3, anh_goc_tt, anh_goc_p1, anh_goc_p2, anh_goc_p3 = cat_cac_phan(
        anh_nhi_phan, anh_goc)

    y, x = anh_nhi_phan.shape
    off_tt_x, off_tt_y = int(x / 1.5), 0
    off_p1_x, off_p1_y = 0, int(y / 3.3)
    off_p2_x, off_p2_y = 0, int(y / 2)
    off_p3_x, off_p3_y = 0, int(y / 1.6)

    sbd, mdt, sbd_positions, mdt_positions = xu_ly_thong_tin(vung_thong_tin, anh_goc_tt, off_tt_x, off_tt_y)
    kq_phan_1 = xu_ly_phan_1(vung_p1, anh_goc_p1, off_p1_x, off_p1_y)
    kq_phan_2 = xu_ly_phan_2(vung_p2, anh_goc_p2, off_p2_x, off_p2_y)
    kq_phan_3 = xu_ly_phan_3(vung_p3, anh_goc_p3, off_p3_x, off_p3_y)

    fc_values = {}
    for i in range(40):
        if i < len(kq_phan_1):
            answers = kq_phan_1[i]
            if len(answers) == 1:
                fc_values[str(i + 1)] = answers
            elif len(answers) > 1:
                fc_values[str(i + 1)] = "Loi"
            else:
                fc_values[str(i + 1)] = []
                if 'info_phan_1' in globals():
                    info = globals()['info_phan_1']
                    khoi_idx = i // 10
                    cau_trong_khoi = i % 10
                    off_x_khoi = 200 + 530 * khoi_idx if info['is_homo'] else info['off_x']
                    m_inv_khoi = info['m_inv'] if info['is_homo'] else (info['cac_minv'][khoi_idx] if khoi_idx < len(info['cac_minv']) else info['m_inv'])
                    ve_gach_ngang_phan_1(cau_trong_khoi, info['is_homo'], m_inv_khoi, off_x_khoi, info['off_y'])
        else:
            fc_values[str(i + 1)] = []

    tf_values = {}
    for i in range(32):
        if i < len(kq_phan_2):
            answers = kq_phan_2[i]
            if len(answers) == 1:
                tf_values[str(i + 1)] = answers
            elif len(answers) > 1:
                tf_values[str(i + 1)] = "Loi"
            else:
                tf_values[str(i + 1)] = []
                if 'info_phan_2' in globals():
                    info = globals()['info_phan_2']
                    khoi_idx = i // 8
                    cau_trong_khoi = i % 8
                    if info['is_homo']:
                        off_x_khoi = 200 + 530 * khoi_idx
                        off_y_khoi = info['off_y']
                        m_inv_khoi = info['m_inv']
                    else:
                        off_x_khoi = info['off_x']
                        off_y_khoi = info['off_y']
                        m_inv_khoi = info['cac_minv'][khoi_idx] if khoi_idx < len(info['cac_minv']) else info['m_inv']
                    ve_gach_ngang_phan_2(cau_trong_khoi, info['is_homo'], m_inv_khoi, off_x_khoi, off_y_khoi)
        else:
            tf_values[str(i + 1)] = []

    dg_values = {}
    for i in range(6):
        if i < len(kq_phan_3):
            item = kq_phan_3[i]
            if isinstance(item, dict):
                value = item.get("value", "")
                valid = item.get("valid", True)
                if not valid:
                    dg_values[str(i + 1)] = "Loi"
                else:
                    dg_values[str(i + 1)] = value
                if value == "" and 'info_phan_3' in globals():
                    info = globals()['info_phan_3']
                    ve_gach_ngang_phan_3(i, info['is_homo'], info['m_inv'], info['off_x'], info['off_y'])
            else:
                dg_values[str(i + 1)] = item if item else ""
        else:
            dg_values[str(i + 1)] = ""

    ket_qua = {
        "org": duong_dan_goc,
        "out": duong_dan_xuat,
        "warn": "",
        "err": [],
        "res": {
            "fc": fc_values,
            "tf": tf_values,
            "dg": dg_values,
        },
        "sbd": sbd,
        "mdt": mdt,
    }

    hien_thi_sbd_mdt(sbd, mdt, sbd_positions, mdt_positions)
    xuat_sbd_mdt_console(sbd, mdt, sbd_positions, mdt_positions)

    os.makedirs('Json', exist_ok=True)
    os.makedirs('Output', exist_ok=True)

    with open(duong_dan_json, "w", encoding="utf-8") as f:
        json.dump(ket_qua, f, indent=2, ensure_ascii=False)

    cv2.imwrite(duong_dan_xuat, anh_xuat)
    print(f"Đã xử lý xong ảnh: {ten_file}")


thu_muc_phieu = 'PhieuQG'
danh_sach_phieu = glob.glob(f'{thu_muc_phieu}/*')

print(f"Đang tiến hành chấm {len(danh_sach_phieu)} phiếu...")
for phieu in danh_sach_phieu:
    cham_phieu(phieu)
print("Đã hoàn thành chấm toàn bộ phiếu!")