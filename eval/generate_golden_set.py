import json
import os

PROBLEMS = [
    ("gsm_01", "Maria has 3 boxes of pencils. Each box has 24 pencils. She gives 18 pencils to her brother. How many pencils does Maria have left?", 54, "arithmetic_multistep"),
    ("gsm_02", "A train travels 60 miles per hour for 3 hours, then 80 miles per hour for 2 hours. How many total miles did the train travel?", 340, "rate_distance"),
    ("gsm_03", "Tom buys 4 shirts at $12 each and 2 pairs of pants at $25 each. He pays with a $150 bill. How much change does he get back?", 52, "money"),
    ("gsm_04", "A bakery makes 144 cupcakes. They sell 3/4 of them in the morning and 1/3 of the remainder in the afternoon. How many cupcakes are left?", 24, "fractions"),
    ("gsm_05", "Sarah is twice as old as her sister Lily. In 5 years, Sarah will be 31. How old is Lily now?", 13, "age_problem"),
    ("gsm_06", "A rectangular garden is 15 meters long and 8 meters wide. If fencing costs $7 per meter, how much will it cost to fence the entire garden?", 322, "geometry"),
    ("gsm_07", "A school has 5 classrooms with 28 students each. If 15 students are absent today, how many students are present?", 125, "arithmetic_multistep"),
    ("gsm_08", "John earns $18 per hour and works 6 hours a day, 5 days a week. How much does he earn in 4 weeks?", 2160, "money_rate"),
    ("gsm_09", "A water tank holds 500 liters. It is currently 40% full. How many more liters are needed to fill it completely?", 300, "percentage"),
    ("gsm_10", "Emma reads 23 pages per day. How many full days will it take her to finish a 300-page book?", 14, "division_remainder"),
    ("gsm_11", "A pizza is cut into 8 slices. 3 friends each eat 2 slices, and a 4th friend eats 1 slice. How many slices are left?", 1, "arithmetic_multistep"),
    ("gsm_12", "A car's fuel tank holds 50 liters. It uses fuel at 8 liters per 100 km. How many km can it travel on a full tank?", 625, "rate_division"),
    ("gsm_13", "There are 240 apples to be packed into boxes of 16. How many boxes are needed, and how many apples are left over if 14 boxes are filled completely first?", 16, "division_remainder"),
    ("gsm_14", "A store marks up the cost of a $40 item by 25% to set the selling price, then offers a 10% discount on the selling price. What is the final price?", 45, "percentage_chain"),
    ("gsm_15", "Two friends start saving money. Friend A saves $15 per week, Friend B saves $22 per week. After how many weeks will Friend B have saved exactly $105 more than Friend A?", 15, "rate_comparison"),
    ("gsm_16", "A farmer has 84 chickens and 36 ducks. He wants to put them into groups so that each group has the same number of chickens and the same number of ducks, with no animals left over, using the maximum possible number of groups. How many groups can he make?", 12, "gcd_grouping"),
    ("gsm_17", "A movie theater sells tickets for $9 for adults and $6 for children. One showing sold 45 adult tickets and 60 child tickets. What was the total revenue?", 765, "money_multistep"),
    ("gsm_18", "A recipe requires 2.5 cups of flour to make 20 cookies. How many cups of flour are needed to make 50 cookies?", 6.25, "proportion"),
    ("gsm_19", "Jake has $200. He spends 1/4 of it on a video game, then 1/2 of what remains on a controller. How much money does he have left?", 75, "fractions_sequential"),
    ("gsm_20", "A factory produces 480 units in an 8-hour shift. If production rate stays constant, how many units would it produce in a 5-hour shift?", 300, "rate_proportion"),
    ("gsm_21", "A library has 1,200 books. 35% are fiction, and the rest are non-fiction. How many non-fiction books are there?", 780, "percentage"),
    ("gsm_22", "Three workers can paint a fence in 12 hours working together at the same rate each. If only 2 of those workers are available, how many hours will it take them to paint the same fence?", 18, "work_rate"),
    ("gsm_23", "A bus has 52 seats. On a trip, 38 seats are occupied. At the next stop, 9 passengers get off and 14 get on. How many seats are now occupied?", 43, "arithmetic_multistep"),
    ("gsm_24", "A number increased by 12 and then doubled gives 96. What is the original number?", 36, "algebra_simple"),
]


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "golden_set.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for pid, question, answer, category in PROBLEMS:
            f.write(json.dumps({
                "id": pid, "question": question, "answer": answer, "category": category
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(PROBLEMS)} problems to {out_path}")


if __name__ == "__main__":
    main()
