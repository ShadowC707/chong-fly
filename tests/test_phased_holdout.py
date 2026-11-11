from evaluate.phased_holdout import admission,longest,cases

def good():
 return dict(collision=None,completed=True,min_clearance_m=.1,near_10cm_longest_s=0.,below_10cm_s=0.,timeout_events=0,stationary_longest_s=1.,distance_m=7.,auto_s=44.,max_height_error_m=.02)

def test_no_collision_does_not_admit_stall_or_near_miss():
 assert admission(good())['passed']
 for k,v in [('min_clearance_m',.049),('near_10cm_longest_s',1.01),('below_10cm_s',2.01),('stationary_longest_s',5.01),('timeout_events',1),('auto_s',0.)]:
  assert not admission(good()|{k:v})['passed']

def test_contiguous_duration_resets_on_clear_sample():
 assert longest([True,True,False,True],[.2,.3,.2,.4])==.5

def test_case_plan_is_fixed_and_mirrored():
 c=cases();assert len(c)==12 and len({v['name'] for v in c})==12
 assert c==cases()
 for a,b in zip(c[8]['boxes'],c[9]['boxes']):
  assert a['x_min']==b['x_min'] and a['y_min']==-b['y_max']
