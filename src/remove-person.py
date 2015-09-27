import webappflask.models as m

tinder_id = '54c2597074c7e43f2525a39e'

person = m.db.session.query(m.Person).filter(m.Person.tinder_id == tinder_id).one()
print(person)
m.db.session.query(m.MeasurementData).filter(m.MeasurementData.person == person).delete()
m.db.session.query(m.Photo).filter(m.Photo.person == person).delete()
m.db.session.delete(person)
m.db.session.commit()
print("success")