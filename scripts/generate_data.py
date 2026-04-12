"""Generate diverse synthetic airline review dataset."""
import sys, random, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

random.seed(42)

airlines = ["United", "Delta", "American Airlines", "Southwest", "JetBlue", "Spirit", "Frontier", "Alaska Airlines"]
cities = ["New York", "LA", "Chicago", "Dallas", "Denver", "Miami", "Seattle", "San Francisco", "Atlanta", "Boston"]
names = ["John", "Sarah", "Mike", "Lisa", "David", "Emma", "Chris", "Anna", "Tom", "Kate"]

def r(*choices): return random.choice(choices)

def gen_negative():
    templates = [
        lambda: f"@{r(*airlines).replace(' ','')} flight from {r(*cities)} to {r(*cities)} was delayed by {random.randint(2,8)} hours. {r('No explanation given.','Staff was unhelpful.','Missed my connection.','This is unacceptable.')}",
        lambda: f"Lost my {r('luggage','bag','suitcase','checked bag')} on {r(*airlines)} flight. {r('Been waiting','No response','Called 3 times','Filed a claim')} for {random.randint(2,14)} days. {r('Still nothing.','No compensation.','Terrible service.','Will never fly again.')}",
        lambda: f"{r(*airlines)} {r('cancelled','canceled')} my flight {r('without notice','last minute','2 hours before departure','with no warning')}. {r('Had to rebook','Spent $'+str(random.randint(200,800))+' on a new ticket','Stranded at the airport','No hotel offered')}.",
        lambda: f"The {r('check-in','boarding','gate')} process at {r(*airlines)} was {r('terrible','a nightmare','chaotic','disorganized')}. {r('Waited','Stood in line','Queued')} for {random.randint(30,120)} minutes. {r('Ridiculous.','Never again.','Worst experience.')}",
        lambda: f"{r(*airlines)} customer service is {r('awful','terrible','non-existent','the worst')}. {r('Called','Been on hold','Tried calling')} {random.randint(2,6)} times. {r('Nobody helped.','Got disconnected.','No resolution.','They hung up on me.')}",
        lambda: f"Seat {r('broken','uncomfortable','dirty','stained')} on {r(*airlines)} flight. {r('No legroom','Food was inedible','Entertainment system broken','AC not working')}. {r('Paid $'+str(random.randint(300,900))+' for this?','Total ripoff.','Disgusting.')}",
        lambda: f"@{r(*airlines).replace(' ','')} your {r('app','website','online system','booking portal')} {r('crashed','is broken','lost my reservation','charged me twice')}. {r('Fix this!','Need a refund.','Unbelievable.','This is fraud.')}",
        lambda: f"{r('Rude','Dismissive','Unprofessional','Condescending')} {r('flight attendant','gate agent','staff member','crew')} on {r(*airlines)}. {r('Refused to help.','Yelled at a passenger.','Ignored my request.','Made me feel unwelcome.')}",
        lambda: f"My {r('connecting','transfer')} flight on {r(*airlines)} was {r('missed','gone','already departed')} because of {r('their delay','late arrival','slow boarding','gate change')}. {r('No rebooking assistance.','Had to pay for hotel.','Lost a full day.')}",
        lambda: f"{r(*airlines)} overbooked the flight and {r('bumped me','denied boarding','removed me','kicked me off')}. {r('Offered only $'+str(random.randint(100,300))+' voucher.','No apology.','This should be illegal.','Worst airline ever.')}",
        lambda: f"The {r('food','meal','snack')} on {r(*airlines)} was {r('disgusting','inedible','stale','cold')}. {r('Even my kids refused to eat it.','Paid extra for this garbage.','How is this acceptable?')}",
        lambda: f"@{r(*airlines).replace(' ','')} {r('why','how come','explain why')} is my refund still not processed after {random.randint(2,8)} weeks? {r('This is theft.','I want my money back.','Filing a complaint with DOT.')}",
    ]
    return random.choice(templates)()

def gen_positive():
    templates = [
        lambda: f"Great experience with {r(*airlines)}! {r('On-time departure','Smooth flight','Early arrival','Perfect landing')} and {r('friendly crew','excellent service','comfortable seats','great entertainment')}. {r('Highly recommend!','Will fly again!','Best airline!','10/10')}",
        lambda: f"{r(*airlines)} {r('customer service','support team','agents')} {r('resolved','fixed','handled')} my issue {r('quickly','in minutes','perfectly','same day')}. {r('Thank you!','Impressed!','Great job!','Keep it up!')}",
        lambda: f"Love flying {r(*airlines)}! {r('Affordable prices','Clean planes','Nice seats','Good food')} and {r('always on time','reliable service','no hassle','smooth experience')}.",
        lambda: f"Just flew {r(*airlines)} from {r(*cities)} to {r(*cities)}. {r('Upgraded to first class!','Got extra legroom.','Best meal I ever had on a plane.','The lounge was amazing.')} {r('Wonderful!','Fantastic service!','A+ experience!')}",
        lambda: f"Shoutout to {r(*names)} at {r(*airlines)} {r(*cities)} for {r('going above and beyond','being so helpful','making my day','excellent service')}. {r('You rock!','Thank you so much!','Best agent ever!')}",
        lambda: f"{r(*airlines)} boarding was {r('fast','efficient','smooth','well-organized')}. {r('Plane was clean.','Crew was welcoming.','Seats were comfortable.')} {r('Nice flight overall!','Would definitely recommend.','Pleasantly surprised!')}",
        lambda: f"My {r('family','kids','parents','group')} had a {r('wonderful','amazing','fantastic','great')} time on {r(*airlines)}. {r('Entertainment kept kids busy.','Food was actually good.','Crew was patient and kind.')}",
        lambda: f"{r(*airlines)} refunded my ticket in {random.randint(1,3)} days. {r('Quick and easy process.','No hassle at all.','Impressed with the speed.','Excellent customer care.')}",
        lambda: f"No delays, no problems, no complaints. {r(*airlines)} from {r(*cities)} was {r('perfect','flawless','exactly what I needed','spot on')}.",
        lambda: f"The flight was fast and there were no delays. {r(*airlines)} {r('nailed it','did great','impressed me','delivered')}.",
        lambda: f"Not a single issue with {r(*airlines)}. {r('Luggage arrived on time.','Boarding was quick.','Crew was professional.','Everything went smoothly.')}",
        lambda: f"No lost baggage, no rude staff, no delays. {r(*airlines)} is doing it right! {r('Highly recommend!','Keep it up!','My go-to airline.')}",
        lambda: f"Flight was not delayed at all. {r(*airlines)} {r('arrived early','landed on time','departed right on schedule')}. {r('Love it!','So refreshing.','Finally a reliable airline.')}",
        lambda: f"Never had a bad experience with {r(*airlines)}. {r('Always on time.','Staff is always friendly.','Prices are fair.')} {r('Top tier!','No complaints here.','Solid airline.')}",
        lambda: f"Can not complain about {r(*airlines)}. {r('The flight was smooth.','Service was good.','Everything was on time.')} {r('Would fly again.','Satisfied customer.','Happy with my choice.')}",
    ]
    return random.choice(templates)()

def gen_neutral():
    templates = [
        lambda: f"{r(*airlines)} flight from {r(*cities)} was {r('okay','fine','average','nothing special')}. {r('On time.','Slight delay.','Standard service.')} {r('Not bad.','Could be better.','Its alright.','Meh.')}",
        lambda: f"Flying {r(*airlines)} {r('today','tomorrow','next week','this weekend')}. {r('Hope its good.','First time.','Lets see.','Fingers crossed.')}",
        lambda: f"{r(*airlines)} is {r('decent','okay','alright','so-so')} for the {r('price','cost','money','fare')}. {r('Nothing fancy but gets you there.','Standard domestic airline.','You get what you pay for.')}",
        lambda: f"Just {r('booked','reserved','bought tickets')} with {r(*airlines)} for {r(*cities)}. {r('Anyone have experience?','Good or bad?','What should I expect?','Tips?')}",
        lambda: f"The {r(*airlines)} {r('app','website')} is {r('working','functional','okay','basic')}. {r('Could use improvement.','Does the job.','Nothing special.','Managed to book my flight.')}",
        lambda: f"{r(*airlines)} lounge at {r(*cities)} airport is {r('decent','okay','small','crowded')}. {r('Food selection is limited.','Chairs are comfortable though.','WiFi works fine.')}",
        lambda: f"Waiting at {r(*cities)} airport for {r(*airlines)} flight. {r('Gate changed once.','Boarding in 30 min.','Standard boarding process.','On time so far.')}",
    ]
    return random.choice(templates)()

# Generate dataset
rows = []
distribution = [("negative", gen_negative, 6400), ("positive", gen_positive, 4200), ("neutral", gen_neutral, 4000)]

for sentiment, generator, count in distribution:
    for _ in range(count):
        text = generator()
        # Add some random noise for uniqueness
        if random.random() < 0.3:
            text += " " + r("smh","ugh","wow","lol","tbh","fyi","imho","seriously","honestly","wtf") if sentiment == "negative" else ""
        if random.random() < 0.2:
            text = text + " #" + r("airlineproblems","flightdelay","customerservice","travel","flying","airline","neveragain","grateful","bestflight")
        rows.append({"airline_sentiment": sentiment, "text": text, "airline": r(*airlines)})

random.shuffle(rows)

# Save
outpath = Path(__file__).parent.parent / "data" / "twitter_airline_sentiment.csv"
outpath.parent.mkdir(parents=True, exist_ok=True)

with open(outpath, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["airline_sentiment", "text", "airline"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} reviews → {outpath}")
from collections import Counter
c = Counter(r["airline_sentiment"] for r in rows)
print(f"Distribution: {dict(c)}")
