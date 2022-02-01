import streamlit as st
from src.model import PoetFormer
from transformers import AutoTokenizer, AutoConfig


@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa")
    config = AutoConfig.from_pretrained("HooshvareLab/gpt2-fa")
    return PoetFormer.load_from_checkpoint('weights/GPT2-fa-ganjoor-conditional/last.ckpt', config=config, tokenizer=tokenizer)

with st.spinner('Loading model...'):
    model = load_model()


all_poets = ['امیرخسرو دهلوی', 'مولانا', 'سعدالدین وراوینی', 'نصرالله منشی', 'فرخی سیستانی', 'شیخ بهایی', 'انوری', 'عمان سامانی', 'مسعود سعد سلمان', 'فروغی بسطامی', 'رشیدالدین میبدی', 'عطار', 'خلیل الله خلیلی', 'ملک الشعرای بهار', 'فیض کاشانی', 'ابن حسام خوسفی', 'پروین اعتصامی', 'رودکی', 'عنصری', 'اوحدی', 'شاه نعمت الله ولی', 'باباافضل کاشانی', 'ازرقی هروی', 'رهی معیری', 'عارف قزوینی', 'امیر معزی', 'همام تبریزی', 'مهستی گنجوی', 'ناصرخسرو', 'حافظ', 'عرفی', 'عبدالقادر گیلانی', 'عبید زاکانی', 'رشحه', 'محتشم کاشانی', 'سلمان ساوجی', 'صائب تبریزی', 'فخرالدین اسعد گرگانی', 'غبار همدانی', 'جامی', 'سعدی', 'شاطرعباس صبوحی', 'اسدی توسی', 'منوچهری', 'شهریار', 'شیخ محمود شبستری', 'عراقی', 'کسایی', 'سیف فرغانی', 'هاتف اصفهانی', 'سنایی', 'فردوسی', 'حزین لاهیجی', 'رشیدالدین وطواط', 'ابوسعید ابوالخیر', 'کمال خجندی', 'کمال الدین اسماعیل', 'وحشی', 'عبدالواسع جبلی', 'ظهیر فاریابی', 'فایز', 'باباطاهر', 'خواجوی کرمانی', 'صامت بروجردی', 'هجویری', 'قدسی مشهدی', 'حکیم نزاری', 'بیدل دهلوی', 'هلالی جغتایی', 'خاقانی', 'اقبال لاهوری', 'قاآنی', 'خیام', 'رضي الدین آرتیمانی']
st.title('Shirin Sokhan, The Poem Generator')
st.subheader("Select a poet to condition the generation.")
poet = st.selectbox('Poet', all_poets, index=1) 

with st.beta_expander("See poet's historgram"):
    st.image('data/poet_hist.png')

st.subheader('Enter the beginning of the poem:')
prompt = st.text_input('Enter text')

temp = st.slider('Temprature', min_value=0.1, max_value=3., value=0.8)
topk = st.slider('Top k', min_value=1, max_value=1000, step=10, value=50)
top_p = st.slider('Top P', min_value=0.1, max_value=1., value=0.9)
n_beam = st.slider('# Beams', min_value=1, max_value=100, step=1, value=1)
max_len = st.slider('Max Length', min_value=30, max_value=1024, step=50)

submit = st.button('Generate!')
if submit:
    st.header('Poem in '+poet+' style!')
    with st.spinner('Generating poem...'):
        poem = model.generate(
            prompt=prompt, 
            poet=poet, 
            num_return_sequences=1, 
            max_length=max_len, 
            n_beam=n_beam, 
            top_p=top_p, 
            topk=topk
        )[0]
        html = "<p style='text-align: right;' dir='rtl'>"+ poem.replace(f'{poet}:\n','').replace('\n', '</br>') + "<p/>"
        st.markdown(html, unsafe_allow_html=True)
        # st.text(poem.replace(f'{poet}:\n',''))
