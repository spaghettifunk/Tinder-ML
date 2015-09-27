__author__ = 'tobias'
import webappflask.models as m

username = 'Constance'

user = m.db.session.query(m.User).filter_by(username=username).one()
print(user)
m.db.session.query(m.MeasurementData).filter(m.MeasurementData.user == user).delete()
m.db.session.delete(user)
m.db.session.commit()
print("success")