from evaluate.sitl_yaw_twr import assess_pair


def test_secondary_recovery_cannot_hide_missed_primary_deadline():
    nominal=dict(final_directed_yaw_deg=17.)
    result=dict(completed=True,final_directed_yaw_deg=17.,max_height_error_m=.02,
        saturated_motor_ticks=0,command_settle_from_step_end_s=.18,
        actual_rate_settle_from_step_end_s=.9,actual_rate_settle_from_profile_end_s=.66)
    verdict=assess_pair(nominal,result)
    assert not verdict['passed'] and not verdict['checks']['physical_recovery']
    assert verdict['physical_recovery_margin_s']<0
    result['actual_rate_settle_from_step_end_s']=None
    assert assess_pair(nominal,result)['physical_recovery_margin_s'] is None


def test_wrong_direction_and_motor_limit_prevent_pass_even_with_fast_recovery():
    nominal=dict(final_directed_yaw_deg=17.)
    result=dict(completed=True,final_directed_yaw_deg=-17.,max_height_error_m=.02,
        saturated_motor_ticks=1,command_settle_from_step_end_s=.18,actual_rate_settle_from_step_end_s=.2)
    verdict=assess_pair(nominal,result)
    assert not verdict['passed']
    assert not verdict['checks']['direction'] and not verdict['checks']['no_motor_limit']
