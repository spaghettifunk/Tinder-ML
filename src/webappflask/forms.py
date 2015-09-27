from flask.ext.wtf import Form
from wtforms import StringField, SubmitField, SelectField, PasswordField
from wtforms.validators import DataRequired

class RegisterForm(Form):
    username = StringField('Your Name', validators=[DataRequired()])
    gender = SelectField('gender', choices=[(1, 'female'), (2, 'male'), (2, "i don't want to say/neither")], coerce=int)
    interested_in = SelectField('Interested in', choices=[(1, 'female'), (2, 'male')], coerce=int)
    submit = SubmitField('Submit')

class PasswordForm(Form):
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Submit')