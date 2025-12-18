import streamlit as st
import joblib
import re
import string
import base64
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ==========================================
# 0️⃣ KONFIGURASI HALAMAN & ENCODER
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen DPR",
    page_icon="🤖",
    layout="centered"
)

# --- FUNGSI UTILITY ---
def update_input_from_selectbox_asli():
    selected_value = st.session_state.selected_sample_asli
    if selected_value != "-- PILIH DATASET ASLI --":
        st.session_state.current_input = selected_value
        st.session_state.selected_sample_baku = "-- PILIH DATASET BAKU --"

def update_input_from_selectbox_baku():
    selected_value = st.session_state.selected_sample_baku
    if selected_value != "-- PILIH DATASET BAKU --":
        st.session_state.current_input = selected_value
        st.session_state.selected_sample_asli = "-- PILIH DATASET ASLI --"

# --- INISIALISASI SESSION STATE ---
if 'current_input' not in st.session_state:
    st.session_state.current_input = ""
# ----------------------------------


def get_base64_of_bin_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Gambar Utama (Background) dan Gambar Tambahan
BG_IMAGE_FILENAME = "gamabr"
EXTRA_IMAGE_FILENAME = "images.jpg"

BG_IMAGE_B64 = get_base64_of_bin_file(BG_IMAGE_FILENAME)
EXTRA_IMAGE_B64 = get_base64_of_bin_file(EXTRA_IMAGE_FILENAME)


# ==========================================
# 1️⃣ CSS STYLE INJECTION
# ==========================================
if BG_IMAGE_B64:
    background_css = f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(10, 25, 47, 0.40), rgba(10, 25, 47, 0.60)),
            repeating-linear-gradient(
                45deg, rgba(100, 255, 218, 0.02), rgba(100, 255, 218, 0.02) 2px, transparent 2px, transparent 40px
            ),
            url("data:image/jpeg;base64,{BG_IMAGE_B64}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
else:
    background_css = """
    <style>
    .stApp {
        background-image: repeating-linear-gradient(45deg, rgba(100, 255, 218, 0.02), rgba(100, 255, 218, 0.02) 2px, transparent 2px, transparent 40px),
            radial-gradient(circle at center, #112240 0%, #0a192f 100%);
    }
    </style>
    """

ui_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; color: #ccd6f6; }

/* Container Utama (Glassmorphism) */
.block-container {
    background-color: rgba(17, 34, 64, 0.2);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 3rem 2rem !important;
    border: 1px solid rgba(100, 255, 218, 0.08);
    max-width: 900px;
}

h1 {
    font-weight: 700;
    background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 5px;
    letter-spacing: 1px;
}

/* Gambar Kiri */
.image-left-style {
    border-radius: 12px;
    overflow: hidden;
    margin-top: 15px;
    border: 3px solid #64ffda;
    box-shadow: 0 0 20px rgba(100, 255, 218, 0.3);
}

.result-container { display: flex; justify-content: center; margin-top: 30px; }

/* Kartu Hasil Sederhana */
.result-card {
    background: rgba(17, 34, 64, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px 25px;
    width: 100%;
    max-width: 400px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.sentiment-badge {
    font-size: 28px;
    font-weight: 700;
    padding: 15px 40px;
    border-radius: 50px;
    display: inline-block;
    color: white;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
/* Tambahan CSS untuk Teks Bersih (dihapus dari output, tapi CSSnya biarkan saja) */
.clean-text-box {
    background-color: rgba(0, 0, 0, 0.2);
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 20px;
    font-family: monospace;
    font-size: 14px;
    color: #64ffda;
    text-align: left;
    word-wrap: break-word;
}
</style>
"""

st.markdown(background_css, unsafe_allow_html=True)
st.markdown(ui_style, unsafe_allow_html=True)


# ==========================================
# 2️⃣ PREPROCESSING & RESOURCE LOADING
# ==========================================
try:
    FACTORY = StemmerFactory()
    STEMMER = FACTORY.create_stemmer()
except:
    STEMMER = None

@st.cache_data
def text_preprocessing(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    if STEMMER:
        text = STEMMER.stem(text)
    return text.strip()

@st.cache_resource
def load_resources():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        models = {
            "Random Forest": joblib.load("model_RF_GamGwo.pkl"),
            "Logistic Regression": joblib.load("model_LR_GamGwo.pkl"),
            "SVM": joblib.load("model_SVM_GamGwo.pkl")
        }
        return vectorizer, models
    except Exception as e:
        st.error(f"Gagal memuat sistem. Pastikan file model dan vectorizer ada di direktori yang benar: {e}")
        return None, None

VECTORIZER, MODELS = load_resources()

# ==========================================
# 3️⃣ KONFIGURASI ALGORITMA SATU (GABUNGAN)
# ==========================================
CHOSEN_MODEL_NAME = "Random Forest"
if MODELS and CHOSEN_MODEL_NAME in MODELS:
    MODEL_TO_USE = MODELS[CHOSEN_MODEL_NAME]
else:
    MODEL_TO_USE = None


# ==========================================
# 3.5️⃣ DAFTAR DATASET (DI BAGI 2 KATEGORI)
# ==========================================
SAMPLE_COMMENTS_ASLI = [
    'Dpr jancok dpr tidak adil dasar',
    'Setuju gaji anggota dewan umr supaya orang tidak ambisisius berlomba lomba untuk menjadi anggota dewan karena tergiur gaji besar',
    'Brukakaka 1000% bayar PBB. Yang tinggal di kolong jembatan layak gk. Bangke bangke asal ngomong aja',
    'Mantap tarian jogetnya. Macam monyet dapat pisang.',
    'Iy ...,tapi kan anggota dewan.,pada orang kayna...beramal sedikit tuk rak yat kan lebih bagus...😮\n,.,',
    'Iy ...,tapi kan anggota dewan.,pada orang kayna...beramal sedikit tuk rak yat kan lebih bagus...😮\n,.,',
    'Apa dpr . Mau jaga rakyat . Atau mau siksa rakyat . Setuju . Nerima uang rakyat . Segitu banyak . Tolong dpr . Janga rampok rakyat',
    'Kadang memang bikin hati panas dan ngerasa nggak adil. Tapi jangan sampai rasa iri itu bikin kita patah semangat ya ✊\nKita tetap jauh lebih berharga walau gaji 3jt per bulan ,karena kerja keras kita halal, nyata, dan hasilnya benar-benar dinikmati keluarga sendiri ❤️',
    'Semoga dapet musibah gaji dpr naik udh enak ksh tunjangan hidup, gak naik juga gajinya dpr tetep gede itu, giliran buruh, kuli, guru, atau dibidang apapun itu malah segitu² aja gak ada kenaikan secara signifikan, gak adil sumpah gak waras negara ini..',
    'Pejabat paling terkorup indonesia.\n-kerja ora iso\n-gajih besar hampir nyamain negara maju\n-pajak di tanggung negara\nHebat kan....',
    'OHHH KITA BAYAR PAJAK DAN SEGALANYA SAMPAI APA APA DI KASIH PAJAK TERUS ADA LAGI BAYAR ROYALTI\n\nITU SEMUA UNTUK KENAIKAN GAJI DPR ? Dengan dalil kompensasi apalah itu halah bahasa halusnya hebat',
    'Hebat ...anggota DPR bisa makmur.. dan suksesss...dan kaya raya....sampai dunia kiamat ...ha ha ha...kok rakyat nonton dan diam aja....ha ha ha......',
    'Kalau perkara rujab gk  layak huni,, kan bisa uang kompensasi nya dijadiin buat renovasi,, lagipula kompensasi  50jt perbulan x 5 tahun,, lah,, itu mah bisa beli 1 rumah  harga 3 M,,',
    'Saya lebih setuju gajih TNI di naikan TNI nyawa taruhannya, TNI Garda terdepan, kalau DPR banyak madhorot nya, habis habisin anggaran negara aja',
    'Mana dulu yang bilang klok probowo jadi presiden indo bakal makmur konyol makan tuh yang dukung Wowo',
    'Rumah ga layak langsung dapet duit...\nApa kabar bagi rakyat yg di wakilkan pada tidur di jalan',
    'BUBARKAN MRP DPR. ALOKASI GAJI MERRKA BUAT RAKYAT, GURU YANG LEBIH BERMANFAAT DIBANDINGKAN MEREKA YG DUDUK DI DPR MPR.\n\nDUKUNG DEMO 25 AGUSTUS',
    'Masih banyak pengangguran, masih ada yang kelaparan ,hebat',
    'Rakyatnya masih kesulitan untuk tempat tidur \nDPR RI bilang mereka tinggal di rumah tidak layak \n\nPejabat negara ini aneh \nIndonesia pasti akan krisis dalam beberapa tahun kedepan \nDan akan terjadi kerusuhan yang besar dan itu sudah pasti \nMasyarakat tidak puas dengan kinerja pejabat negara',
    'Ora usah nggo mumet...golput adalah jalan terbaik 👍👍👍👍👍🇮🇩🇮🇩🇮🇩🇮🇩🇮🇩',
    'Memang indonesia hebat betul y, rakyat pasti setuju tu, KLO gaji DPR di naik kan, apa lah arti ny kami rakyat ini, ya kan boss?',
    'Naik gaji sampe 100jt trs...tp ada yg blg GURU ADALAH BEBAN NEGARA...APA GAK SALAH.??\nINI YG COCOK JD BEBAN NEGARA',
    'Rakyat.kelaparan.pak tolong kasih pekerjaan.yg layak.',
    'Rakyat.kelaparan.pak tolong kasih pekerjaan.yg layak.',
    'Mantap lanjutkan',
    'Apa sih fungsi DPR MPR? Yang di rasakan adalah mereka seperti preman berkumpul di satu tempat.\n\nTidak ada rakyat yang benar-benar di wakili oleh mereka. Laknat lah kalian pemakaman uang rakyat secara zalim lewat jalur Pajak. Semoga kelak perut kalian kenyang akan minum air nanah didalam nerakah Jahanam.\n\nRakyat di zalimi dengan kebijakan, lapor Kapolri hanya berakhir dengan penjarah beberapa hari karena hukum di Indonesia adalah hukum yang tidak pernah adil. Maka mari kita laporkan mereka kepada Allah SWT atas kebijakannya hingga tak ada ampunan mereka dan mendekam lah mereka di dasar Jahanam yang panas dan mendidih.',
    'Koplkkkk. Mau tunjang rumah tunjangan apapun tidak adil buat rakyat kecil. Kompen sasi rumah buat rakyat 15 jt.  Itu juga berupa barang akhit nya gak tuntas karna modelnya rmhnua dah di atur. Dan harus nombok..lieuuuurrrrr',
    'SI EKO PATRIO DAN GEROMBOLANYA ANGGOTA DPR PALING KOPLAK PASTI RAKYAT YG MEMILIH DIA NYESEL SEGEDE GUNUNG.  GAK ADA DPR NEGARA TETAP JALAN ASAL ORANGNYA BENER, BUBARKAN DPR..\nSETUJU👍👍😁',
    'Tunjangan rumah kata mereka yaaaaa terserahlaaaa,kalian berjoget kami nenontonkalian. Semoga yg maha adil allah swt melaknat kalian.para pemimpin yang tidak adil aamiin yrbl almin',
    'Ga adil klu cuman anggota DPR yg naik guru tuh harusnya yg dinaikan mana janjinya pak presiden naikan gaji guru katanya.🙏',
    'Yang setuju revolusi ,mari kita suarakan revolusi kita ulangan tragedi 97 hanya iti jalan keluaranya',
    'Tunggu saja Revolusi itu akan terjadi ada waktu dan saatnya Rakyat Indonesia bergerak. Nikmatilh waktu kalian sebelum waktu itu habis dan datang. \n" LUPA BERKACA 98 "\nApa kabar Jendral koalisi koruptor janji masih ingat ? Rakyat Indonesia yang akan mengingatkan jika lupa\nBaru kali ini bangsa Indonesia salah memilih presiden karena merasa kasihan dan benar  hasilnya 0 besar. \nAntri Gas Lpg hingga mat*\nMinyak langkah beras oplosan\nPupuk susah\nPengangguran menurun ekonomi Indonesia tumbuh kata siapa ? \nKoruptor bebas\nRekening dan Tanah nganggur dapat disita\nGaji DPR 100JT \nGaji Guru ? 300rb/bulan\nAnak presiden terbaik Indonesia Gibran hanya duduk lemas miris sedih saat melihat koalisi koruptor omon omon berjoget dengan disahkannya gaji 100JT. \nTernyata Tukang Kayu lebih baik dan berani tidak pernah tunduk dengan mak lampir dan koalisi koruptor. \n\n2029 atau sebelumnya wajib ganti presiden.',
    'Gajih tidak naik aja orang masih banyak yang ingin jadi DPR apa lagi naik??\nJustru harus nya tunjangan nya di turunin agar jdi dpr ikhlas demi rakyat bukan demi gajih\nYg setuju like',
    'Rakyat kaya DPR miskin',
    'Kami rakyat miskin sangat setuju kalau ģaji dpr sama dgn gaji pns.',
    'Joget2 gaji naik, sangat menyakitkan rakyat miskin dan sulit cari makan, sdg mereka di gaji   dg uang rakyat.😮',
    'begini kah hidup di indonesia dan begini kah yang diingin kan oleh para leluhur warga menangis karena kemeskinan sedangkan dpr yang sudah kaya tambah kaya dan koruptor di negara ini semakin banyak apa kah benar yang dibilang oleh prabowo bahwa indonesia pada tahun 2030 akan bubar???',
    'jangan makan uang rakyat mikirin orang-orang yang nggak mampu yang miskin yang mulung dan lain-lain jangan mikirin uang pribadi mikirin orang-orang di bawah jangan makan uang haram',
    'Cokk aku sebagai rakyat tidak ridho gaji anggota dewan dinaikkan begitu juga konpensasi rumah juga dinaikkan saya tidak ridho dunia akhirat, rumah anggota dewan bagus bagus kaya² seperti Bramasta, eko patrio harusnya yang dijadikan anggota dewan itu orang miskin insyaallah jujur baik akhlaknya tidak semena mena.',
    'yg kaya makin kaya yg miskin makin miskin',
    'Aku org Malaysia ketar ketir....😊😊😊 Gji DPR anggota Parlimen dibyar ratusan Juta RP....Gaji Guru hnya dibyar 300 Ribu..😢😢😢...Guru beban ngra bilangnya... Memang ngra yg x pya rasa malu...TKI yg kerja potong sawit pn di Malaysia dibayar gaji tinggi ..',
    'tenyata ge baru tau arti APK (Anjayyy Pemerintah Korupsi)',
    'CUIHHH NAJIS JOGED2 DI ATAS PENDERITAAN RAKYAT,,NEGARA BOBROK MIRIS MIRISS...',
    'Pejabat makin kaya, Masyarakatnya mkin pada miskin.. Ampun parah ya Allah😔',
    'tidak punya malu,  di saat rakyat menderita di cekik bayar pajak, mereka malah joget² dapat kenaikan gaji... terlaknat kalian semua yang menari nari di atas penderitaan rakyat',
    'itulah anggota dpr pintar bersilat lidah, gaji di ganti nama menjadi tunjangan... banyak rakyat yg sehari makan cuma 1x bahkan ada yg tidak makan seharian, ini malah wakil nya joget2 senang dpt tunjangan 100jt/ bulan...\nnegeri apa coba ini... negara maju saja anggota dpr nya naik angkutan umum bukan naik alphard eh ini negara miskin anggota dpr nya ga ada yg naik angkutan umum...',
    'Bikin miskin rakyat aja n dewan...',
    'Kami rakyat sangat kecewa sama DPR  semoga pemilu berikutnya nggak ada yg milih mereka, semoga rakyat segera sadar...',
    'Pikiran jelek sy bnr di naikan gaji dngn atas namakan kopensasi UANG rumah tinggl . Karna skrang kan segala bhn bakar mahal semua apa lagi bhn bakar buat makan',
    'rakyat menangis,mbok ya lebih memikirkan nasib rakyat dari kalangan menengah k bawah ibu /bpk yg terhormat,masih  sangat bnyak rakyat2 miskin kekurangan pangan,tidak ada tempat tinggal,kurang perhatian aparat tentang kesehatan,baru2 ini kasus almarhumah adek raya yg dari kluarga tidak mampu,harus khilangannyawa karena ribuan cacing bersarang d tubuh kecilny dan karena kurang perhatian dari aparat desa,menyedihkan sekali,..\ntolonglah pak/ibu lihat mereka yg sangat mmbutuhkan,bpk/ibu anggota DPRD yg bisa mkn enak,bisa tidur  d tempat nyaman,tolong lihatlah mereka2 ini yg mkn pun susah walau cuma bisa mkn nasi saja sudah alhamdulillah,BPK/ibu apa tega menerima gaji yg besar sedangkan rakyat menangis,sedangkan gaji beliau jg berasal dari rakyat sedangkan rakyat menderita😢',
    'Yang kaya makin kaya..yang miskin makin miskin',
    'Rombongan orang miskin, telor bensin rumah beras, minta ditunjangan, yang bayar rakyat lewat pajak, sedangkan lo semua nga kena pajak penghasil gw sebagai rakyat nga iklhas lahir batin semoga lo semua dapat balasan yang setimpal, aamiin',
    'kenaikan tunjangan anggota dpr dibayar dengan menaikan pajak yg semakin mencekik rakyat...... BIADAP MUKA2 ANGGOTA DPR RI GAK PUNYA MALU',
    'RAKYAT SANGAT KESULITAN BUAT BELI BERAS, EHH ANGOTA DPR MALAH DAPAT TUNJANGAN BERAS,  SANGAT MENYEDIHKAN. RAKYAT BEGITU MISKINNYA. MALAH TUNJANGAN BERAS NYA LUAR BIASA, ANGOTA DPR ADA SUAMI ISTRI. BAYANGKAN TUNJANGAN BERAS. TUNGAN RUMAH. TUNJANGAN LISTRIK. TUNJANGAN SALON.  POKOKNYA PULUHAN TUNJANGAN. TUNJANGAN TIAP RAPAT. TUNJANGAN KE DAERAH. BELUM LAGI KELUAR NEGERU.  POKOKNYA SEMUA KERJA CUMAN URUSAN DUIT. JADI RAKYAT INDONESI YANG BISA DIKATAKAN LEBIH SEPAROH MISKIN , JANGAN BERHARAP SEJAHTRA.',
    'Yg kaya makin kaya yg miskin makin miskin itulah negara kita Indonesa karna harta kekayaan negri ini hanya d rasakan para elit anggota dewan rayat sisa g mungkin rayat bisa subur makmur klo para pejabatnya g mau Deket sama rakyatnya',
    'Sementara rakyat sibuk tiap hari dari pagi sampai malam peras keringat banting tulang cuma buat mencari sesuap nasi.\n\nitu rakyat menengah ke bawah Rakyat miskin lagi sedih Sedihnya berjuang terhadap hidupnya terhadap keluarganya Sedangkan anda joget joget di gedung sana.\n\nKetidakpedulian yang dirasakan oleh masyarakat yan, erjuang untuk memenuhi kebutuhan dasar hidupnya, sementara pihak lain terlihat menikmati kemewahan dan kesenangan tanpa memikirkan kesulitan rakyat.\n\nKita bubarin aja dpr masak kita satu indonesia kalah sama satu gedung.\n\nIndonesia tanpa DPR akan tetap baik-baik saja, camkan itu',
    'Mental bobrok pada di jadikan anggota dewan\n...lebih tepatnya mereka itu cuma makan gajih buta\n..percumah negara menggelontorkan dana tiap bulan buat ngasih gajih ke mereka.....',
    'Bubar kn DPR penipu ranya ngk ada gunanya',
    'Negara konoha benar benar negara bobrok yang kerja berat  keringat sampe kelubang pantat gaji cuman sekian puluh ribu perak',
    'DPR enak pa naik gaji saya warga miskin kesusahan jualan aja sepi barang sembako pada naik ,',
    'Gaji banyak insaalloh musibanya jugak banya soalnya menyangkut masarakat banyak ygsengsara orang miskin pasti doanya dikabulkan oleh alloh amin',
    'Yang terekam lagi joged harusnya malu sak malu2nya.. dilihat anak, keluarga, kerabat dan tetangga dirumah..',
    'Wah tambah enak hidup ny PR wakil rkyt, hidupny sngt trjamin, yg ky mkn ky, yg miskin  kckik msalh ekonomi yg smkin srba mhl, utang negara bnyk bangt, wahai PR penguasa negri Indonesia jdlh pr pemimpin yg bijaksana,brnts PR korupsi dngn bnr',
    'Suatu malam, Umar bin Khattab r.a. keluar sendirian untuk memantau keadaan rakyatnya. Ia melihat ada seorang ibu bersama anak-anaknya di dalam tenda, mereka kelaparan. Sang ibu merebus air, pura-pura memasak agar anak-anaknya tenang dan tertidur, padahal sebenarnya tidak ada makanan.\n\nMelihat hal itu, Umar menangis dan segera kembali ke Baitul Mal. Ia meminta penjaga mengambilkan gandum dan minyak. Sang penjaga menawarkan untuk mengangkatkan barang itu, tetapi Umar menolak dan berkata:\n\n> “Apakah engkau mau memikul dosaku di hari kiamat?”\n\n\n\nMaka Umar sendiri yang memikul karung gandum itu di punggungnya. Ia membawanya ke tenda ibu tadi, lalu memasakkan sendiri makanan untuk keluarga miskin itu hingga anak-anak kenyang dan tertawa. Umar duduk menunggu sampai wajah mereka berseri-seri, barulah ia pulang dengan tenang.\n\nKami butuh wakil rakyat yang punya Mental seperti itu, tapi sayang, yg kita dapat malah sebaliknya,.. "wakil" yang tidak mewakili rakyat,.. 😑, bahkan cenderung mementingkan kebutuhan dan keinginan mereka sendiri, ya Allah, perilaku mereka menunjukan seolah mereka tidak takut akan apa yang kelak menimpa mereka di hari perhitungan',
    'Menari di atas penderitaan rakyat miskin',
    'Fungsi DPR apa sih . Masyarakat diluaran sana masih bnyak yg kekurangan dan harusnya pejabat malu rakyat nya kebanyakan memilih jd babu dinegara org pdhal Indonesia itu kaya akan sumber alam .',
    'DPR, Pemerintah dan Lembaga yang menaikkan gaji anggota DPR, harusnya menaikkan taraf hidup rakyat, bukan menaikkan taraf hidup para anggota DPR yang sebelumnya sudah kaya raya !',
    'Bagaimana cara nyiksa mereka biar menyesal!!! ⁉️',
    'Indopiece Real😂',
    'gedung DPR tempatnya tikus tikus berdasi cocoknya di kasih king kobra untuk berburu tikus',
    'Pilih ustadz Adi Hidayat ❤🎉',
    '.menggeramkan',
    'Anggota dpr harus diperiksa kejiwaan nya,,',
    'Ayo di demo lagi, itu tunjangan perumahan yg 50 jt bukan di hilangin tapi di pindahin ke tunjangan reses jadi tunjangan reses naik 50 juta, coba cari beritanya pasti ketemu...emg brengsek mereka itu, gk kapok juga di demo kemaren',
    'ih geramnyo \nDPR Jelek',
    'Ooo dasar dpr gembel minta gaji dari pajak rakyat gak ngotak eeee dpr eek',
    'Aku bukan bela DPR yah, tapi aku cuma ngingetin cnn',
    'Video joget itu kan setelah rapat tahunan, tapi kalian malah sandingkan video itu dalam pembahasan kenaikan gaji DPR, itu bisa buat orang beranggapan bahwa DPR joget karna merayakan naik gaji, lain kali fikir dulu sebelum uplod',
    'Jika pembahasan kalian tentang gaji anggota dewan, itu bertolak belakang dengan video Anggita dewan yang joged, nanti orang malah mikir\n: DPR joged merayakan gajian,\nNah itu video kalian bahaya gimana sih',
    'Hey cnn apakah kalian tidak sadar, dengan kesalahan kalian yang bikin demo kemarin',
    'Dpr anj',
    'Kerja seumur hidup gaji kecil tanpa pensiun, kerja 5 tahun ngantor saat ada rapat ada pensiun, itulah kesenjangan honorer pppk dibandingkan dengan DPR, kenapa DPR bisa dapat pensiun dan PPPK tidak dapat, karena yang membuat undang undang itu dewan😂😂😂, dan bahkan seandainya diperbolehkan ,DPR akan membuat aturan saat rapat DPR jumlah berapa kali anggota dpr kencing ke kamar mandi dapat tunjangan😂😂😂😂, misal rapat 4 jam, pergi kencing setiap 2 jam maka 2 kencing x 400 rb 😂😂😂😂😂😂😂😂',
    'Hartati pak ,,,,,,,,rekening ready,,,,dana ,,,,,top ,,,,,,boleh cekgo',
    'GAJI DPR HARUS UMR LAH YG GAJI MRK ITU KAN dari PAJAK RAKYAT. Bisa mikir gak sih....',
    'AndaikAn gaji PNS dan dpr serta karyawan di negeri ini paling tinggi 30 jt dan terkecil 10 jt maka rakyatnya insyaallah akan makmur sejahtera',
    'kenapa pajak kita paling tinggi dari negara lain???karena buat bayarin monyet2 dpr yg pengen gaji tinggi,yg aturannya dibuat sendiri😅😅',
    'Harus di batalkan tunjangan apa pun, masyarakat di bawah masih ada yang hanya minum air putih, gak bisa beli beras',
    'D P R❗🙌\nDewan Perwakilan Rakyat😮❓\nWakil rakyat Naik mercy😳\nRakyatnya jalan kaki🤕\nWakil rakyat makan sate🍢😋\nRakyat nya makan tempe🧆😪\nIni yang dinamakan❓🙌\nMERDEKA TAPI BINGUNG🥶❗\nINI YANG Dinamakan❗🙌\nBINGUNG TAPI MERDEKAA❗❗',
    'Nanggung jendral,kasih mereka 1 M per bulan biar aman jabatan anda..."pemerintahan yg lucu"!!!',
    'Dasar DPR tlol itu yang kami bgo jangan sok lihat in dari pemerintah itu uang kita e',
    'Bagi la rakyat kau bernafas pulak...kesian la dieorg',
    'puan tuh hanya pandai bicara',
    'Di Indonesia itu pengenya gaji DPR disetarakan dgn parlement di eropa yg bergaji ($/euro) mangkannya hinggah seratus juta ketemunya.\nPadahal mereka ajah gak mampu untuk menurunkan nilai tukar ($/euro) ke Rp seperti zaman Pak Habibi. 😂 gini kok nyebut rakyat yg mau bubarin DPR itu t*l*l.\nKalok gak mampu turunkan nilai tukar maunya bergaji setara dgn eropa tetap bubarin ajah.',
    'alibi tunjangan, bagi2 hasil korup biar tutup mulut',
    '𝙼𝚊𝚔𝚊𝚗𝚢𝚊 𝚝𝚛𝚊𝚗𝚜𝚙𝚊𝚛𝚊𝚗, 𝚍𝚒𝚜𝚒𝚊𝚛𝚔𝚊𝚗 𝚕𝚒𝚟𝚎',
    'Terlaluan klu begitu.',
    'yg joget2 pecatin smua',
    '😂😂😂😂😂 ngakak gila',
    'Puan bau tanah',
    'Pantasan Rakyat pada marah kaya gini 😭😭😭'
]

SAMPLE_COMMENTS_BAKU = [
    'Kebijakan kenaikan tunjangan anggota DPR harus mempertimbangkan kondisi ekonomi masyarakat yang sedang sulit.',
    'Transparansi anggaran dalam pengalokasian dana fasilitas perumahan anggota dewan sangat diperlukan publik.',
    'Seharusnya pemerintah lebih mengutamakan peningkatan kesejahteraan guru honorer daripada tunjangan legislatif.',
    'Efisiensi anggaran negara merupakan hal krusial untuk menjaga stabilitas ekonomi nasional saat ini.',
    'Masyarakat menaruh harapan besar agar anggota DPR menolak fasilitas mewah demi rasa empati sosial.',
    'Pemberian kompensasi perumahan bagi anggota dewan perlu dievaluasi secara mendalam dan terbuka.',
    'Kualitas legislasi seharusnya meningkat seiring dengan bertambahnya fasilitas yang diberikan negara.',
    'Integritas wakil rakyat diuji melalui keberanian mereka dalam membatasi pengeluaran anggaran yang berlebihan.',
    'Keadilan sosial harus menjadi landasan utama dalam setiap penetapan gaji pejabat pemerintah.',
    'Publik menilai bahwa prioritas nasional saat ini adalah pemulihan ekonomi, bukan penambahan gaji pejabat.',
    'Anggota legislatif harus memberikan teladan dalam pola hidup sederhana sebagai bentuk pengabdian bangsa.',
    'Audit independen terhadap penggunaan tunjangan DPR diperlukan untuk menjamin akuntabilitas dana publik.',
    'Kenaikan biaya hidup masyarakat saat ini membuat kebijakan kenaikan gaji DPR menjadi isu yang sensitif.',
    'Keseimbangan antara hak dan kewajiban pejabat negara harus dijaga sesuai dengan norma kepantasan.',
    'Keputusan yang diambil di parlemen harus mencerminkan aspirasi rakyat secara murni dan jujur.',
    'Pemerataan pembangunan di daerah lebih mendesak dibandingkan penambahan fasilitas di ibu kota.',
    'Sistem pengupahan yang adil bagi seluruh aparatur negara akan menciptakan keharmonisan sosial.',
    'Setiap rupiah dari pajak rakyat harus dikelola dengan prinsip kehati-hatian dan tanggung jawab tinggi.',
    'Evaluasi kinerja anggota dewan secara objektif harus menjadi syarat sebelum adanya kenaikan insentif.',
    'Legislator diharapkan fokus pada penyelesaian undang-undang yang pro-rakyat kecil dan usaha mikro.',
    'Ketidakpekaan sosial dalam pengambilan keputusan anggaran dapat menurunkan kepercayaan publik kepada institusi.',
    'Reformasi birokrasi di lembaga legislatif harus menyasar pada efektivitas penggunaan anggaran reses.',
    'Komitmen terhadap pemberantasan korupsi harus dimulai dari transparansi pendapatan para pejabat.',
    'Rakyat membutuhkan bukti nyata pengabdian wakilnya melalui kebijakan yang meringankan beban hidup.',
    'Stabilitas politik nasional sangat bergantung pada kepuasan masyarakat terhadap keadilan distribusi anggaran.',
    'Pengalihan rumah dinas menjadi tunjangan tunai berisiko menciptakan pemborosan jika tidak diawasi ketat.',
    'Kepentingan nasional wajib ditempatkan di atas kepentingan kelompok politik manapun di parlemen.',
    'Moralitas politik menuntut para pemimpin untuk selalu bersikap bijaksana dalam menggunakan fasilitas negara.',
    'Pemberdayaan ekonomi rakyat harus menjadi fokus utama dalam setiap nota keuangan pemerintah.',
    'DPR sebagai lembaga representatif harus lebih proaktif dalam mendengarkan kritik konstruktif masyarakat.',
    'Peningkatan fasilitas pejabat tanpa dibarengi hasil kerja nyata akan menimbulkan sentimen negatif berkelanjutan.',
    'Negara harus menjamin bahwa dana publik dialokasikan untuk sektor kesehatan dan pendidikan berkualitas.',
    'Pejabat publik yang berintegritas tinggi akan selalu mendahulukan kepentingan umum daripada materi pribadi.',
    'Rasionalisasi anggaran di semua lembaga tinggi negara adalah langkah tepat di masa pemulihan ekonomi.',
    'Kedaulatan rakyat tecermin dari kebijakan anggaran yang memperhatikan nasib warga di garis kemiskinan.',
    'Publik menghargai anggota dewan yang berani menyuarakan penolakan terhadap pemborosan keuangan negara.',
    'Kepantasan sosiologis harus menjadi pertimbangan dalam menentukan besaran tunjangan bagi pejabat.',
    'Fungsi pengawasan DPR harus diperkuat untuk memastikan setiap kebijakan pemerintah tepat sasaran.',
    'Kesenjangan pendapatan antara pejabat dan rakyat jelata tidak boleh dibiarkan semakin melebar.',
    'Dialog terbuka antara DPR dan elemen masyarakat diperlukan sebelum mengesahkan kenaikan anggaran internal.',
    'Pembangunan infrastruktur di pelosok negeri membutuhkan dukungan pendanaan yang sangat besar.',
    'Ketegasan dalam memotong anggaran yang tidak produktif menunjukkan kualitas kepemimpinan yang baik.',
    'Sinergi antara pemerintah dan DPR dalam penghematan energi dan biaya operasional harus ditingkatkan.',
    'Aspirasi masyarakat terkait penolakan tunjangan perumahan merupakan bentuk kepedulian warga negara.',
    'Akuntabilitas moral adalah beban yang harus dipikul oleh setiap individu yang duduk di kursi parlemen.',
    'Standardisasi gaji pejabat publik harus mengacu pada kemampuan fiskal negara secara jangka panjang.',
    'Rakyat menghendaki adanya pengawasan ketat terhadap setiap aliran dana tunjangan anggota legislatif.',
    'Kepuasan publik terhadap kinerja parlemen merupakan indikator keberhasilan demokrasi di Indonesia.',
    'Pejabat negara harus memiliki rasa tanggung jawab yang besar dalam mengelola harta kekayaan negara.',
    'Pengabdian tulus tanpa mengharapkan kemewahan adalah ciri negarawan sejati yang dirindukan rakyat.',
    'Kesejahteraan masyarakat pedesaan harus menjadi target utama dalam setiap kebijakan fiskal nasional.',
    'Setiap anggota dewan harus mempertanggungjawabkan setiap dana yang diterima kepada konstituennya.',
    'Kebijakan yang memihak pada kepentingan elit politik akan mencederai semangat reformasi bangsa.',
    'Efisiensi belanja pegawai merupakan langkah strategis untuk mengurangi defisit anggaran pendapatan negara.',
    'Transformasi digital di parlemen seharusnya mampu mengurangi biaya operasional yang tidak perlu.',
    'Keadilan bagi tenaga honorer harus diperjuangkan sejajar dengan pembahasan tunjangan pejabat.',
    'Institusi DPR harus menjadi simbol kesederhanaan dan kerja keras bagi seluruh rakyat Indonesia.',
    'Masa depan bangsa ditentukan oleh keberanian pemimpin dalam mengambil keputusan yang adil dan jujur.',
    'Pertumbuhan ekonomi yang inklusif hanya bisa dicapai melalui tata kelola keuangan yang transparan.',
    'Rakyat akan selalu mendukung kebijakan pemerintah yang benar-benar berfokus pada kebutuhan dasar warga.',
    'Kritik terhadap kenaikan tunjangan adalah bagian dari hak demokrasi yang dijamin oleh undang-undang.',
    'Integritas dan dedikasi harus menjadi landasan utama bagi setiap penyelenggara negara di Indonesia.',
    'Pemerintah perlu meninjau kembali urgensi pemberian fasilitas tambahan bagi pejabat di masa krisis.',
    'Kesadaran kolektif untuk melakukan penghematan anggaran harus dimulai dari jajaran pimpinan tertinggi.',
    'Distribusi pendapatan yang merata akan menjamin stabilitas keamanan dan ketertiban masyarakat.',
    'Program jaminan sosial bagi rakyat miskin lebih mendesak untuk didanai daripada tunjangan mewah.',
    'Keputusan politik yang bijak adalah yang mampu menyentuh hati rakyat dan memberikan solusi nyata.',
    'Setiap undang-undang yang disahkan harus memberikan dampak positif bagi kemajuan ekonomi rakyat.',
    'Transparansi publik dalam hal gaji pejabat akan mengurangi potensi penyalahgunaan wewenang.',
    'Sikap empati terhadap kesulitan warga adalah modal utama bagi seorang wakil rakyat yang amanah.',
    'Keberlanjahan fiskal negara sangat bergantung pada ketepatan dalam menentukan prioritas belanja.',
    'Rakyat menuntut adanya efektivitas kerja yang nyata dari seluruh anggota dewan yang terhormat.',
    'Peningkatan taraf hidup masyarakat bawah harus menjadi misi utama setiap wakil rakyat di parlemen.',
    'Ketimpangan sosial adalah musuh demokrasi yang harus dilawan dengan kebijakan anggaran yang adil.',
    'Negara harus hadir dalam memberikan perlindungan ekonomi bagi setiap warga negara tanpa kecuali.',
    'DPR diharapkan menjadi jembatan aspirasi yang kokoh antara rakyat dan pemerintah pusat.',
    'Etika politik melarang penggunaan anggaran negara untuk kepentingan kemewahan pribadi pejabat.',
    'Penyederhanaan birokrasi dan anggaran adalah kunci efisiensi dalam pemerintahan modern.',
    'Rakyat akan menghormati pemimpin yang berani hidup sederhana di tengah kesulitan bangsanya.',
    'Kebijakan anggaran yang pro-rakyat akan meningkatkan martabat bangsa di mata internasional.',
    'Setiap anggota legislatif wajib menjunjung tinggi nilai-nilai kejujuran dalam mengelola dana negara.',
    'Keberhasilan pembangunan nasional diukur dari berkurangnya angka kemiskinan dan pengangguran.',
    'DPR harus menjadi contoh dalam implementasi tata kelola lembaga yang bersih dan akuntabel.',
    'Masyarakat berharap adanya reformasi dalam sistem pemberian tunjangan bagi seluruh pejabat negara.',
    'Kekuatan sebuah negara terletak pada kepercayaan rakyatnya terhadap para pemimpin di parlemen.',
    'Sikap kritis masyarakat adalah pendorong utama bagi perbaikan kinerja institusi pemerintahan.',
    'Penghapusan fasilitas yang berlebihan merupakan langkah nyata dalam mendukung penghematan nasional.',
    'Pembangunan sumber daya manusia lebih penting daripada pemenuhan fasilitas mewah elit politik.',
    'Pemerintah dan DPR harus bekerja sama dalam menciptakan kebijakan fiskal yang sehat dan kuat.',
    'Amanah rakyat harus dijalankan dengan penuh rasa syukur dan dedikasi tinggi bagi nusa dan bangsa.',
    'Setiap kebijakan ekonomi harus diarahkan pada terciptanya lapangan kerja baru bagi generasi muda.',
    'Kepedulian terhadap lingkungan dan masalah sosial harus tecermin dalam anggaran pembangunan.',
    'DPR wajib memastikan bahwa pajak yang dibayarkan rakyat kembali dalam bentuk pelayanan publik berkualitas.',
    'Moralitas dan etika harus selalu mendahului kepentingan politik dalam setiap rapat paripurna.',
    'Kepercayaan publik adalah aset yang sangat mahal dan harus dijaga dengan kinerja yang nyata.',
    'Sistem demokrasi akan berjalan baik jika ada transparansi penuh dalam penggunaan dana negara.',
    'Kenaikan tunjangan tidak boleh menjadi alasan bagi penurunan semangat pengabdian kepada rakyat.',
    'Rakyat merindukan sosok pemimpin yang lebih banyak bekerja daripada menuntut fasilitas tambahan.',
    'Indonesia yang maju hanya bisa terwujud jika seluruh elemen bangsa memiliki semangat keadilan sosial.',
    'Semoga setiap keputusan yang diambil di gedung dewan selalu mendapatkan rida dari Tuhan Yang Maha Esa.'
]

# ==========================================
# 4️⃣ FUNGSI KOREKSI MANUAL
# ==========================================
def force_correct_prediction(clean_text: str, prediction: str) -> str:
    """
    Fungsi ini memeriksa keberadaan kata-kata negatif yang sangat kuat 
    (setelah distemming) dan mengoreksi prediksi menjadi 'negatif' 
    jika model ML memberikan hasil yang salah (misalnya Positif).
    """
    STRONG_NEGATIVE_KEYWORDS = ['buruk', 'jelek', 'bobrok', 'korup', 'bobrol', 'salah', 'tolak', 'gagal', 'miskin']
    if prediction.lower() == 'positif':
        if any(keyword in clean_text for keyword in STRONG_NEGATIVE_KEYWORDS):
            return 'Negatif (Koreksi Manual)'
    
    return prediction

# ==========================================
# 5️⃣ TAMPILAN UTAMA & LOGIKA PREDIKSI
# ==========================================
st.markdown("<h1>ANALISIS SENTIMEN AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi Opini Publik Isu Gaji DPR | Optimasi GAM-GWO</div>", unsafe_allow_html=True)

if VECTORIZER is None or MODEL_TO_USE is None:
    st.error("⚠️ Sistem gagal dimuat atau model tidak ditemukan.")
    st.stop()

# --- BLOK CONTOH KATA (Utuh Sesuai Aslinya) ---
positive_words = ["mantap", "bagus", "sukses", "hebat", "terbaik", "cocok", "adil", "bijak", "bersyukur"]
negative_words = ["tolak", "gagal", "rugi", "miskin", "korupsi", "mahal", "bodoh", "malu", "kecewa"]
neutral_words = ["rapat", "usulan", "pimpinan", "komisi", "kebijakan", "anggaran", "membahas", "jakarta", "sidang"]
# ---------------------------------------------


# Layout Input
with st.container():
    col_img, col_input = st.columns([1, 2])

    with col_img:
        if EXTRA_IMAGE_B64:
             st.markdown('<div class="image-left-style">', unsafe_allow_html=True)
             st.image(f"data:image/jpeg;base64,{EXTRA_IMAGE_B64}", use_column_width=True)
             st.markdown('</div>', unsafe_allow_html=True)
        else:
             st.info("Gambar tambahan tidak ditemukan.")

    with col_input:
        # Pilihan Kategori 1: Dataset Asli
        st.markdown("<p style='font-weight: 600; margin-bottom: 5px;'>1. Pilih dari Dataset Asli (Campuran):</p>", unsafe_allow_html=True)
        st.selectbox(
            label="Komentar Sampel Asli",
            options=["-- PILIH DATASET ASLI --"] + SAMPLE_COMMENTS_ASLI,
            key="selected_sample_asli",
            on_change=update_input_from_selectbox_asli,
            label_visibility="collapsed"
        )

        # Pilihan Kategori 2: Dataset Baku
        st.markdown("<p style='font-weight: 600; margin-top: 10px; margin-bottom: 5px;'>2. Pilih dari Dataset Baku (Formal):</p>", unsafe_allow_html=True)
        st.selectbox(
            label="Komentar Sampel Baku",
            options=["-- PILIH DATASET BAKU --"] + SAMPLE_COMMENTS_BAKU,
            key="selected_sample_baku",
            on_change=update_input_from_selectbox_baku,
            label_visibility="collapsed"
        )

        # Pilihan Input Manual
        st.markdown("<p style='font-weight: 600; margin-top: 15px; margin-bottom: 5px;'>Atau Ketik/Edit Komentar di Sini:</p>", unsafe_allow_html=True)
        input_text = st.text_area(
            label="Ketik Komentar",
            value=st.session_state.current_input, 
            placeholder="Ketik komentar di sini...",
            height=100,
            key="current_input",
            label_visibility="collapsed"
        )

        analyze_button = st.button("🔍 ANALISIS SEKARANG")

# Logika Hasil
if analyze_button:
    if st.session_state.current_input.strip() == "":
        st.warning("⚠️ Harap masukkan teks komentar!")
    else:
        # 1. Proses Prediksi ML
        clean_text = text_preprocessing(st.session_state.current_input)

        if VECTORIZER is None or MODEL_TO_USE is None:
            st.error("⚠️ Analisis gagal: Model atau Vectorizer tidak tersedia.")
            st.stop()

        X = VECTORIZER.transform([clean_text])
        ml_prediction = MODEL_TO_USE.predict(X)[0]
        
        # 2. Koreksi Manual (untuk memperbaiki bias model)
        final_prediction = force_correct_prediction(clean_text, ml_prediction)
        
        # 3. Styling Hasil
        if 'koreksi' in final_prediction.lower() or final_prediction.lower() == "negatif":
            label = "NEGATIF"
            badge_bg = "linear-gradient(90deg, #dc2626, #f87171)"
            icon = "😡"
        elif final_prediction.lower() == "positif":
            label = "POSITIF"
            badge_bg = "linear-gradient(90deg, #059669, #34d399)"
            icon = "😊"
        else:
            label = "NETRAL"
            badge_bg = "linear-gradient(90deg, #64748b, #94a3b8)"
            icon = "😐"

        # Tampilkan Kartu Hasil
        st.markdown(f"""
        <div class="result-container">
            <div class="result-card">
                <h4 style="color: #ccd6f6; margin-bottom: 5px;">HASIL ANALISIS SENTIMEN</h4>
                <div class="sentiment-badge" style="background: {badge_bg};">
                    {icon} &nbsp; {label}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
