from env.event_calendar import EventCalendar
def test_cal():
 c=EventCalendar();c.push(2,"a");c.push(1,"b");assert c.pop()[1]=="b"
