from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(data):
    doc = SimpleDocTemplate("report.pdf")

    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Autism Screening Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Diagnosis: {data['diagnosis']}", styles['Normal']))
    content.append(Paragraph(f"CARS Score: {round(data['cars_score'],2)}", styles['Normal']))
    content.append(Paragraph(f"Eye Contact: {data['eye_percent']}%", styles['Normal']))

    content.append(Spacer(1, 20))
    content.append(Paragraph("Detailed Category Scores:", styles['Heading2']))

    for k, v in data["categories"].items():
        content.append(Paragraph(f"{k}: {round(v,2)}", styles['Normal']))

    doc.build(content)