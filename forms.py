from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField

class UploadImageForm(FlaskForm):
    picture = FileField('Upload A Picture', validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField('Upload')

class UploadImagesForm(FlaskForm):
    picture1 = FileField('Upload A Picture', validators=[FileAllowed(['jpg', 'png'])])
    picture2 = FileField('Upload A Picture', validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField('Upload')

class UploadVideoForm(FlaskForm):
    file = FileField('Upload A Video', validators=[FileAllowed(['mp4', 'avi'])])
    submit = SubmitField('Upload')
    
# class UploadImage2Form(FlaskForm):
#     picture = FileField('Upload A Picture', validators=[FileAllowed(['jpg', 'png'])])
#     submit = SubmitField('Upload')