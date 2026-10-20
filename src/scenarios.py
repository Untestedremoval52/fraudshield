def reason_text(feature, tx):
    if feature.startswith("channel_"):
        return f"Payment channel: {feature.split('_', 1)[1]}"
    elif feature.startswith("merchant_category_"):
        return f"Payment category: {feature.split('_', 2)[2]}"
    fan_in = tx.get("ben_fan_in", 0)
    texts = {"ben_fan_in" : f"The receiving account got money from {fan_in} different accounts recently forming a typical mule pattern." if fan_in > 2 else "The receiving account is brand new, with almost no history.", "is_new_beneficiary" : "This is the first-ever payment to this account.",
             "amount_ratio" : f"{tx.get('amount_ratio', 1):.1f}x this customer's usual amount.", "active_call" : "A phone call was active during the payment.", 
             "screen_share" : "Screen sharing was switched on during the call.", "new_app_installed" : "A new app was installed just before.",
             "otp_fill_seconds" : f"An OTP was entered in {tx.get('otp_fill_seconds', 0):.1f} seconds, which is faster than a person's typing speed.", "mins_since_login" : f"The payment was made {tx.get('mins_since_login', 0):.1f} minutes after the login was made.",
             "hour" : f"The payment was made at {tx.get('hour', 0)}:00, outside the usual hours.", "is_night" : "The payment was made at night.",
             "is_new_city" : "The payment was made from a city this customer has never used.", "is_new_device" : "The payment was made from a device this customer has never used.",
             "is_new_channel" : "The payment was made using a method this customer has never used.", "failed_attempts" : f"There were {tx.get('failed_attempts', 0)} failed attempts before this payment.",
             "amount" : "Unusual amount"}
    return texts.get(feature, feature)