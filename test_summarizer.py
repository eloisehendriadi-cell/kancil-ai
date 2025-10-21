from the_summarizer import summarize_text

long_text = """
Raja Ampat, located in Indonesia's West Papua province, is considered one of the most biodiverse marine regions on Earth.
It consists of over 1,500 small islands and is home to more than 1,600 species of reef fish and 600 species of hard coral.
The coral reef ecosystem here is vital not only for marine biodiversity but also for local communities who depend on fishing and ecotourism for survival.
However, this delicate environment is increasingly threatened by overfishing, tourism, and climate change.
Efforts have been made by local governments and NGOs to create marine protected areas (MPAs), engage local communities in conservation, and promote sustainable tourism practices.
These steps are crucial to ensuring the long-term survival of Raja Ampat's unique ecosystem, as well as the livelihoods of the people who call it home.
The region also plays a significant role in carbon sequestration thanks to its vast mangrove forests, which absorb large amounts of CO2.
This makes it not only a biodiversity hotspot, but also an essential piece of the global climate puzzle.
Preserving Raja Ampat is more than just a matter of environmental protection—it is an investment in the future health of our planet.
""" * 5  # Multiply to simulate a long input

summary = summarize_text(long_text)
print("📝 Summary:\n")
print(summary)

