async function uploadAndExtract() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        alert('الرجاء اختيار ملف أولاً');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    alert('جاري تحليل المستند واستخراج الأسئلة وتوليد الخيارات، قد يستغرق ذلك بضع ثوانٍ...');

    try {
        let response = await fetch('http://127.0.0.1:5000/api/generate-quiz', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('فشل عملية التحليل من السيرفر');

        let data = await response.json();
        localStorage.setItem('quizData', JSON.stringify(data));
        window.location.href = 'quiz.html';
    } catch (error) {
        console.error('Error:', error);
        alert('حدث خطأ أثناء الاتصال بالسيرفر أو تحليل الملف');
    }
}