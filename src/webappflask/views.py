import os
from .config import app, photodir
from flask import session, request, redirect, url_for, render_template, send_from_directory
from .forms import RegisterForm, PasswordForm
from sqlalchemy.sql.expression import func
from sqlalchemy import not_
from .models import (
    db,
    User,
    MeasurementData,
    Photo,
    Person,
)
from sqlalchemy.sql.expression import func
from flask.ext.bootstrap import Bootstrap

bootstrap = Bootstrap(app)

def get_session_id():
    query = db.session.query(func.max(MeasurementData.session).label("max_id"))
    res = query.one()
    if res.max_id is None:
        return 1
    else:
        return res.max_id + 1

def get_person_to_rate(userid):
    user = db.session.query(User).filter_by(id=userid).one()
    measured = db.session.query(MeasurementData.person_id).filter_by(user=user).subquery()
    unmeasured_person = db.session.query(Person).outerjoin(measured, measured.c.person_id == Person.id).filter(measured.c.person_id == None).filter(Person.gender == user.likes_gender).order_by(func.random()).first()
    # unmeasured_person = db.session.query(Person).first()
    imagelist = []
    if unmeasured_person is not None:
        images = db.session.query(Photo).filter(Photo.person == unmeasured_person).all()
        for i in images:
            imagelist.append(i.filepath)

        return unmeasured_person.id, imagelist
    else:
        return None, None

@app.route('/')
def home():
    return redirect(url_for('password'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('password_ok') == True:
        username = None
        form = RegisterForm()
        if form.validate_on_submit():
            username = form.username.data
            interested_in = form.interested_in.data
            gender = form.gender.data
            user = User(username, gender, interested_in)
            db.session.add(user)
            db.session.commit()
            db_user = db.session.query(User).filter_by(username=user.username).one()
            app.logger.debug("user-id: {}".format(db_user.id))
            session['user'] = db_user.id
            session['registered'] = True
            return redirect(url_for('rate_person'))
        return render_template('register.html', form=form)
    else:
        return redirect(url_for('password'))

@app.route('/password', methods=['GET', 'POST'])
def password():
    form = PasswordForm()
    if form.validate_on_submit():
        password = form.password.data
        if password == app.config.get('APP_PASSWORD'):
            app.logger.debug("password ok")
            session['password_ok'] = True
            return redirect(url_for('register'))
        else:
            app.logger.debug("entered password: {}, password: {}".format(password, app.config.get('APP_PASSWORD')))
            return redirect(url_for('password'))
    else:
        return render_template('password.html', form=form)

@app.route('/rate-person', methods=['GET', 'POST'])
def rate_person():
    if 'registered' in session:
        if request.method == 'POST':
            if 'like' in request.form:
                # app.logger.debug("session-id: {}".format(session.get('session_id')))
                m = MeasurementData(session.get('session_id'), True)
                m.user = db.session.query(User).filter_by(id=session.get('user')).one()
                m.person = db.session.query(Person).filter_by(id=request.form['person_id']).one()
                app.logger.debug("session-id: {}, like: {}, user: {}, person: {}".format(m.session, m.like, m.user, m.person))
                db.session.add(m)
                db.session.commit()
                return redirect(url_for('rate_person'))
            elif 'dont_like' in request.form:
                # app.logger.debug("session-id: {}".format(session.get('session_id')))
                m = MeasurementData(session.get('session_id'), False)
                m.user = db.session.query(User).filter_by(id=session.get('user')).one()
                m.person = db.session.query(Person).filter_by(id=request.form['person_id']).one()
                app.logger.debug("session-id: {}, like: {}, user: {}, person: {}".format(m.session, m.like, m.user, m.person))
                db.session.add(m)
                db.session.commit()
                return redirect(url_for('rate_person'))
            else:
                return redirect(url_for('rate_person'))
        else:
            session_id = get_session_id()
            user_id = session.get('user')
            person_id, imagelist = get_person_to_rate(user_id)
            if person_id is None:
                return redirect(url_for('thankyou'))
            session['session_id'] = session_id
            app.logger.debug("session_id: {}, person_id:{}, imagelist:{}".format(session_id, person_id, imagelist))
            return render_template('rate-person.html', person_id=person_id, num_pic=0, num_all_pic=0,
                                            images=imagelist, photodir=photodir)

    else:
        return redirect(url_for('register'))

@app.route('/thankyou')
def thankyou():
    return render_template('thank-you.html')

@app.route('/photos/<path:filename>')
def photos(filename):
    return send_from_directory(app.root_path + '/photos/', filename)